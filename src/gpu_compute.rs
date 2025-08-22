use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{RenderApp, Render, ExtractSchedule};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bytemuck::{Pod, Zeroable};
use crate::Velocity;

// GPU-compatible particle data structure
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuParticle {
    pub position: [f32; 3],
    pub _padding1: f32,
    pub velocity: [f32; 3],
    pub _padding2: f32,
    pub radius: f32,
    pub mass: f32,
    pub _padding3: [f32; 2],
}

impl Default for GpuParticle {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            _padding1: 0.0,
            velocity: [0.0; 3],
            _padding2: 0.0,
            radius: 0.05,
            mass: 1.0,
            _padding3: [0.0; 2],
        }
    }
}

// GPU compute uniforms
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct ComputeUniforms {
    pub delta_time: f32,
    pub gravity: f32,
    pub particle_count: u32,
    pub ground_level: f32,
    pub player_position: [f32; 3],
    pub player_radius: f32,
    pub player_velocity: [f32; 3],
    pub _padding: f32,
}

// Resources for GPU compute
#[derive(Resource)]
pub struct GpuComputeResources {
    pub particle_buffer: Buffer,
    pub uniform_buffer: Buffer,
    pub staging_buffer: Buffer,
    pub bind_group: BindGroup,
    pub compute_pipeline: ComputePipeline,
}

// Resource to track GPU particle data - needs to be extractable to render world
#[derive(Resource, Default, Clone)]
pub struct GpuParticleData {
    pub particles: Vec<GpuParticle>,
    pub uniforms: ComputeUniforms,
    pub needs_update: bool,
    pub particle_count: usize,
}

impl ExtractResource for GpuParticleData {
    type Source = GpuParticleData;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

// Resource to hold GPU computation results for main app
#[derive(Resource, Default, Clone)]
pub struct GpuComputeResults {
    pub updated_particles: Vec<GpuParticle>,
    pub ready: bool,
}

impl ExtractResource for GpuComputeResults {
    type Source = GpuComputeResults;

    fn extract_resource(source: &Self::Source) -> Self {
        // Always extract a fresh copy, but preserve the ready state from render app
        source.clone()
    }
}

// Plugin for GPU compute
pub struct GpuComputePlugin;

impl Plugin for GpuComputePlugin {
    fn build(&self, app: &mut App) {
        // Add resources to main app
        app.init_resource::<GpuParticleData>()
            .init_resource::<GpuComputeResults>();
        
        // Add resource extraction plugins
        app.add_plugins(ExtractResourcePlugin::<GpuParticleData>::default())
            .add_plugins(ExtractResourcePlugin::<GpuComputeResults>::default());
        
        // Add systems to main app for data preparation and result application
        app.add_systems(Update, (
            prepare_gpu_particle_data,
            update_gpu_uniforms,
            apply_gpu_results_to_transforms,
        ));
        
        // Add systems to the render app
        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            (
                setup_gpu_compute.run_if(not(resource_exists::<GpuComputeResources>)),
                upload_particle_data.run_if(resource_exists::<GpuComputeResources>),
                upload_uniforms.run_if(resource_exists::<GpuComputeResources>),
                dispatch_compute_shader.run_if(resource_exists::<GpuComputeResources>),
                read_back_particle_data.run_if(resource_exists::<GpuComputeResources>),
            ).chain(),
        );
    }
}

// System to set up GPU compute resources
fn setup_gpu_compute(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    _render_queue: Res<RenderQueue>,
) {
    info!("Setting up GPU compute resources...");
    
    // Create particle buffer
    let particle_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Particle Buffer"),
        size: (std::mem::size_of::<GpuParticle>() * 1000000) as u64, // 1M particles max
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create uniform buffer
    let uniform_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Compute Uniforms"),
        size: std::mem::size_of::<ComputeUniforms>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create staging buffer for GPU-to-CPU readback
    let staging_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Particle Staging Buffer"),
        size: (std::mem::size_of::<GpuParticle>() * 1000000) as u64, // 1M particles max
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Create compute shader
    let shader = unsafe {
        render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Particle Compute Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/particle_compute.wgsl").into()),
        })
    };

    // Create bind group layout
    let bind_group_layout = render_device.create_bind_group_layout(
        Some("Compute Bind Group Layout"),
        &[
            // Particle buffer
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Uniform buffer
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    );

    // Create bind group
    let bind_group = render_device.create_bind_group(
        Some("Compute Bind Group"),
        &bind_group_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: particle_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    );

    // Create compute pipeline layout
    let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create compute pipeline
    let compute_pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("Particle Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Store resources
    commands.insert_resource(GpuComputeResources {
        particle_buffer,
        uniform_buffer,
        staging_buffer,
        bind_group,
        compute_pipeline,
    });

    info!("GPU compute resources initialized successfully!");
}

// System to dispatch the compute shader
fn dispatch_compute_shader(
    gpu_resources: Res<GpuComputeResources>,
    gpu_data: Res<GpuParticleData>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if gpu_data.particle_count == 0 {
        return;
    }

    // Create command encoder
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Particle Compute Encoder"),
    });

    // Begin compute pass
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Particle Compute Pass"),
            timestamp_writes: None,
        });

        // Set pipeline and bind group
        compute_pass.set_pipeline(&gpu_resources.compute_pipeline);
        compute_pass.set_bind_group(0, &gpu_resources.bind_group, &[]);

        // Dispatch compute shader (64 threads per workgroup, adjust based on particle count)
        let workgroup_count = (gpu_data.particle_count as u32 + 63) / 64; // Round up division
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        
        info!("Dispatched GPU compute: {} particles, {} workgroups",
              gpu_data.particle_count, workgroup_count);
    }

    // Copy particle buffer to staging buffer for readback
    let particle_buffer_size = (std::mem::size_of::<GpuParticle>() * gpu_data.particle_count) as u64;
    encoder.copy_buffer_to_buffer(
        &gpu_resources.particle_buffer,
        0,
        &gpu_resources.staging_buffer,
        0,
        particle_buffer_size,
    );

    // Submit commands
    render_queue.submit(std::iter::once(encoder.finish()));
}

// System to prepare GPU particle data from CPU particles
fn prepare_gpu_particle_data(
    particle_query: Query<(&Transform, &Velocity, &crate::RegolithParticle), Without<crate::Player>>,
    mut gpu_data: ResMut<GpuParticleData>,
) {
    // Convert CPU particles to GPU format
    gpu_data.particles.clear();
    
    for (transform, velocity, particle) in particle_query.iter() {
        gpu_data.particles.push(GpuParticle {
            position: [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ],
            _padding1: 0.0,
            velocity: [velocity.0.x, velocity.0.y, velocity.0.z],
            _padding2: 0.0,
            radius: particle.radius,
            mass: particle.mass,
            _padding3: [0.0, 0.0],
        });
    }
    
    gpu_data.particle_count = gpu_data.particles.len();
    gpu_data.needs_update = true;
}

// System to update GPU uniforms
fn update_gpu_uniforms(
    player_query: Query<(&Transform, &Velocity), With<crate::Player>>,
    time: Res<Time>,
    mut gpu_data: ResMut<GpuParticleData>,
) {
    if let Ok((player_transform, player_velocity)) = player_query.single() {
        // Update uniforms in the GPU data resource
        gpu_data.uniforms = ComputeUniforms {
            delta_time: time.delta_secs(),
            gravity: crate::LUNAR_GRAVITY,
            particle_count: gpu_data.particle_count as u32,
            ground_level: crate::MIN_PARTICLE_RADIUS,
            player_position: [
                player_transform.translation.x,
                player_transform.translation.y,
                player_transform.translation.z,
            ],
            player_radius: 0.8, // Match the CPU version
            player_velocity: [
                player_velocity.0.x,
                player_velocity.0.y,
                player_velocity.0.z,
            ],
            _padding: 0.0,
        };
        gpu_data.needs_update = true;
    }
}

// System to upload particle data to GPU
fn upload_particle_data(
    gpu_resources: Res<GpuComputeResources>,
    gpu_data: Res<GpuParticleData>,
    render_queue: Res<RenderQueue>,
) {
    if gpu_data.needs_update && !gpu_data.particles.is_empty() {
        // Upload particle data to GPU buffer
        let particle_data = bytemuck::cast_slice(&gpu_data.particles);
        render_queue.write_buffer(
            &gpu_resources.particle_buffer,
            0,
            particle_data,
        );
        info!("Uploaded {} particles to GPU buffer", gpu_data.particles.len());
    }
}

// System to upload uniforms to GPU
fn upload_uniforms(
    gpu_resources: Res<GpuComputeResources>,
    gpu_data: Res<GpuParticleData>,
    render_queue: Res<RenderQueue>,
) {
    if gpu_data.needs_update {
        // Upload uniform data to GPU buffer
        let uniforms_array = [gpu_data.uniforms];
        let uniform_data = bytemuck::cast_slice(&uniforms_array);
        render_queue.write_buffer(
            &gpu_resources.uniform_buffer,
            0,
            uniform_data,
        );
        info!("Uploaded uniforms to GPU: dt={:.4}, particles={}",
              gpu_data.uniforms.delta_time, gpu_data.uniforms.particle_count);
    }
}

// System to read back particle data from GPU
fn read_back_particle_data(
    gpu_resources: Res<GpuComputeResources>,
    gpu_data: Res<GpuParticleData>,
    render_device: Res<RenderDevice>,
    mut gpu_results: ResMut<GpuComputeResults>,
) {
    if gpu_data.particle_count == 0 {
        return;
    }

    // TODO: Implement actual GPU buffer readback using staging buffer
    // For now, we'll simulate the GPU compute shader results on CPU
    // This allows us to test player interactions while we work on the full GPU pipeline
    
    let mut updated_particles = gpu_data.particles.clone();
    let mut collision_count = 0;
    
    // Simulate the GPU compute shader logic on CPU
    for particle in &mut updated_particles {
        // Apply gravity (matching GPU shader)
        particle.velocity[1] += crate::LUNAR_GRAVITY * gpu_data.uniforms.delta_time;
        
        // Update position
        particle.position[0] += particle.velocity[0] * gpu_data.uniforms.delta_time;
        particle.position[1] += particle.velocity[1] * gpu_data.uniforms.delta_time;
        particle.position[2] += particle.velocity[2] * gpu_data.uniforms.delta_time;
        
        // Ground collision (matching GPU shader)
        if particle.position[1] <= particle.radius {
            particle.position[1] = particle.radius;
            particle.velocity[1] = particle.velocity[1].abs() * -0.2; // RESTITUTION
            particle.velocity[0] *= 0.95; // FRICTION
            particle.velocity[2] *= 0.95; // FRICTION
        }
        
        // Player collision (matching GPU shader logic)
        let player_pos = [
            gpu_data.uniforms.player_position[0],
            gpu_data.uniforms.player_position[1],
            gpu_data.uniforms.player_position[2]
        ];
        
        let dx = particle.position[0] - player_pos[0];
        let dy = particle.position[1] - player_pos[1];
        let dz = particle.position[2] - player_pos[2];
        let player_distance = (dx * dx + dy * dy + dz * dz).sqrt();
        let min_distance = particle.radius + gpu_data.uniforms.player_radius;
        
        if player_distance < min_distance && player_distance > 0.0 {
            collision_count += 1;
            
            // Calculate collision normal
            let normal = [dx / player_distance, dy / player_distance, dz / player_distance];
            
            // Separate particle from player
            let overlap = min_distance - player_distance;
            particle.position[0] += normal[0] * overlap * 0.8;
            particle.position[1] += normal[1] * overlap * 0.8;
            particle.position[2] += normal[2] * overlap * 0.8;
            
            // Debug output for first few collisions
            if collision_count <= 3 {
                info!("Player-particle collision {}: distance={:.3}, overlap={:.3}, particle_pos=[{:.2}, {:.2}, {:.2}]",
                      collision_count, player_distance, overlap,
                      particle.position[0], particle.position[1], particle.position[2]);
            }
            
            // Apply player momentum transfer
            let player_speed = (gpu_data.uniforms.player_velocity[0] * gpu_data.uniforms.player_velocity[0] +
                               gpu_data.uniforms.player_velocity[1] * gpu_data.uniforms.player_velocity[1] +
                               gpu_data.uniforms.player_velocity[2] * gpu_data.uniforms.player_velocity[2]).sqrt();
            
            if player_speed > 0.1 {
                let push_force_scale = player_speed * 0.5;
                let player_vel_len = player_speed;
                particle.velocity[0] += (gpu_data.uniforms.player_velocity[0] / player_vel_len) * push_force_scale;
                particle.velocity[1] += (gpu_data.uniforms.player_velocity[1] / player_vel_len) * push_force_scale;
                particle.velocity[2] += (gpu_data.uniforms.player_velocity[2] / player_vel_len) * push_force_scale;
            } else {
                // Minimum push force
                particle.velocity[0] += normal[0] * 2.0;
                particle.velocity[1] += normal[1] * 2.0;
                particle.velocity[2] += normal[2] * 2.0;
            }
            
            // Collision response
            let rel_vel_dot_normal = particle.velocity[0] * normal[0] +
                                   particle.velocity[1] * normal[1] +
                                   particle.velocity[2] * normal[2];
            
            if rel_vel_dot_normal < 0.0 {
                let impulse_scalar = -(1.0 + 0.3) * rel_vel_dot_normal;
                particle.velocity[0] += normal[0] * impulse_scalar * 0.8;
                particle.velocity[1] += normal[1] * impulse_scalar * 0.8;
                particle.velocity[2] += normal[2] * impulse_scalar * 0.8;
            }
        }
        
        // Apply velocity damping (matching GPU shader)
        particle.velocity[0] *= 0.95; // VELOCITY_DAMPING
        particle.velocity[1] *= 0.95;
        particle.velocity[2] *= 0.95;
        
        // Clamp very small velocities to zero
        let vel_magnitude = (particle.velocity[0] * particle.velocity[0] +
                           particle.velocity[1] * particle.velocity[1] +
                           particle.velocity[2] * particle.velocity[2]).sqrt();
        if vel_magnitude < 0.08 { // MIN_VELOCITY
            particle.velocity[0] = 0.0;
            particle.velocity[1] = 0.0;
            particle.velocity[2] = 0.0;
        }
    }
    
    // Store results
    gpu_results.updated_particles = updated_particles;
    gpu_results.ready = true;
    
    // Log collision summary if any occurred
    if collision_count > 0 {
        info!("Frame summary: {} player-particle collisions detected", collision_count);
    }
    
    info!("Read back {} particles from GPU (CPU simulated with player interactions)", gpu_data.particle_count);
}

// System to apply GPU results to Transform components (runs in main app)
fn apply_gpu_results_to_transforms(
    mut particle_query: Query<(&mut Transform, &mut Velocity), (With<crate::RegolithParticle>, Without<crate::Player>)>,
    gpu_data: Res<GpuParticleData>,
    time: Res<Time>,
    player_query: Query<(&Transform, &Velocity), (With<crate::Player>, Without<crate::RegolithParticle>)>,
) {
    if gpu_data.particle_count == 0 {
        return;
    }
    
    // Get player data for interactions
    let (player_transform, player_velocity) = if let Ok(player_data) = player_query.single() {
        (player_data.0, player_data.1)
    } else {
        return;
    };
    
    // Apply physics directly in main app (temporary solution until GPU readback works)
    let mut applied_count = 0;
    let mut debug_count = 0;
    
    for (mut transform, mut velocity) in particle_query.iter_mut() {
        let old_pos = transform.translation;
        
        // Apply gravity
        velocity.0.y += crate::LUNAR_GRAVITY * time.delta_secs();
        
        // Update position
        transform.translation += velocity.0 * time.delta_secs();
        
        // Ground collision
        let radius = 0.05; // Default particle radius
        if transform.translation.y <= radius {
            transform.translation.y = radius;
            velocity.0.y = velocity.0.y.abs() * -0.2; // Bounce with energy loss
            velocity.0.x *= 0.95; // Friction
            velocity.0.z *= 0.95; // Friction
        }
        
        // Player collision
        let player_distance = transform.translation.distance(player_transform.translation);
        let min_distance = radius + 0.8; // player radius
        
        if player_distance < min_distance && player_distance > 0.0 {
            // Calculate collision normal
            let normal = (transform.translation - player_transform.translation).normalize();
            
            // Separate particle from player
            let overlap = min_distance - player_distance;
            transform.translation += normal * overlap * 0.8;
            
            // Apply player momentum transfer
            let player_speed = player_velocity.0.length();
            if player_speed > 0.1 {
                let push_force = player_velocity.0.normalize() * (player_speed * 0.5);
                velocity.0 += push_force;
            } else {
                // Minimum push force
                velocity.0 += normal * 2.0;
            }
            
            // Collision response
            let relative_velocity = velocity.0;
            let velocity_along_normal = relative_velocity.dot(normal);
            
            if velocity_along_normal < 0.0 {
                let impulse_scalar = -(1.0 + 0.3) * velocity_along_normal;
                let impulse = normal * impulse_scalar;
                velocity.0 += impulse * 0.8;
            }
        }
        
        // Apply velocity damping
        velocity.0 *= 0.95;
        
        // Clamp very small velocities to zero
        if velocity.0.length() < 0.08 {
            velocity.0 = Vec3::ZERO;
        }
        
        applied_count += 1;
        
        // Debug only first few particles to avoid log spam
        if debug_count < 3 {
            info!("Particle {}: {:?} -> {:?}", debug_count + 1, old_pos, transform.translation);
            debug_count += 1;
        }
    }
    
    if applied_count > 0 {
        info!("Applied physics with player interactions to {} particles", applied_count);
    }
}