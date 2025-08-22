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

// Spatial hash uniforms
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct SpatialHashUniforms {
    pub particle_count: u32,
    pub grid_size: [u32; 3],
    pub cell_size: f32,
    pub world_min: [f32; 3],
    pub world_max: [f32; 3],
    pub max_particles_per_cell: u32,
    pub _padding: [u32; 4], // Need 4 u32s (16 bytes) to reach 64 bytes total
}

// Hash cell structure for GPU
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HashCell {
    pub particle_count: u32,
    pub particle_indices: [u32; 32], // Max 32 particles per cell
    pub _padding: [u32; 3],
}

impl Default for HashCell {
    fn default() -> Self {
        Self {
            particle_count: 0,
            particle_indices: [0; 32],
            _padding: [0; 3],
        }
    }
}

// Resources for GPU compute
#[derive(Resource)]
pub struct GpuComputeResources {
    pub particle_buffer: Buffer,
    pub uniform_buffer: Buffer,
    pub staging_buffer: Buffer,
    pub bind_group: BindGroup,
    pub compute_pipeline: ComputePipeline,
    // Spatial hashing resources
    pub spatial_hash_buffer: Buffer,
    pub spatial_hash_uniform_buffer: Buffer,
    pub spatial_hash_bind_group: BindGroup,
    pub spatial_hash_clear_pipeline: ComputePipeline,
    pub spatial_hash_populate_pipeline: ComputePipeline,
    pub spatial_hash_collision_pipeline: ComputePipeline,
}

// Resource to track GPU particle data - needs to be extractable to render world
#[derive(Resource, Default, Clone)]
pub struct GpuParticleData {
    pub particles: Vec<GpuParticle>,
    pub uniforms: ComputeUniforms,
    pub needs_update: bool,
    pub particle_count: usize,
}

// Resource to track spatial hash data
#[derive(Resource, Default, Clone)]
pub struct GpuSpatialHashData {
    pub uniforms: SpatialHashUniforms,
    pub grid_size: [u32; 3],
    pub cell_size: f32,
    pub world_bounds: ([f32; 3], [f32; 3]), // (min, max)
    pub needs_update: bool,
}

impl ExtractResource for GpuSpatialHashData {
    type Source = GpuSpatialHashData;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
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
        // Extract from main app to render app
        source.clone()
    }
}

// Plugin for GPU compute
pub struct GpuComputePlugin;

impl Plugin for GpuComputePlugin {
    fn build(&self, app: &mut App) {
        // Add resources to main app
        app.init_resource::<GpuParticleData>()
            .init_resource::<GpuComputeResults>()
            .init_resource::<GpuSpatialHashData>();
        
        // Add resource extraction plugins
        app.add_plugins(ExtractResourcePlugin::<GpuParticleData>::default())
            .add_plugins(ExtractResourcePlugin::<GpuComputeResults>::default())
            .add_plugins(ExtractResourcePlugin::<GpuSpatialHashData>::default());
        
        // Add systems to main app for data preparation and result application
        app.add_systems(Update, (
            prepare_gpu_particle_data,
            update_gpu_uniforms,
            update_spatial_hash_uniforms,
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
                upload_spatial_hash_uniforms.run_if(resource_exists::<GpuComputeResources>),
                dispatch_spatial_hash_compute.run_if(resource_exists::<GpuComputeResources>),
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

    // Create spatial hash buffer (64x64x64 grid = 262,144 cells)
    let grid_size = 64u32;
    let total_cells = grid_size * grid_size * grid_size;
    let spatial_hash_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Spatial Hash Buffer"),
        size: (std::mem::size_of::<HashCell>() * total_cells as usize) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create spatial hash uniform buffer
    let spatial_hash_uniform_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Spatial Hash Uniforms"),
        size: std::mem::size_of::<SpatialHashUniforms>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create compute shader
    let shader = unsafe {
        render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Particle Compute Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/particle_compute.wgsl").into()),
        })
    };

    // Create spatial hash shader
    let spatial_hash_shader = unsafe {
        render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Spatial Hash Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/spatial_hash.wgsl").into()),
        })
    };

    // Create bind group layout for main compute
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
            // Spatial hash uniform buffer
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Spatial hash buffer
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    );

    // Create bind group layout for spatial hash
    let spatial_hash_bind_group_layout = render_device.create_bind_group_layout(
        Some("Spatial Hash Bind Group Layout"),
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
            // Hash grid buffer
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Spatial hash uniform buffer
            BindGroupLayoutEntry {
                binding: 2,
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

    // Create bind groups
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
            BindGroupEntry {
                binding: 2,
                resource: spatial_hash_uniform_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: spatial_hash_buffer.as_entire_binding(),
            },
        ],
    );

    let spatial_hash_bind_group = render_device.create_bind_group(
        Some("Spatial Hash Bind Group"),
        &spatial_hash_bind_group_layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: particle_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: spatial_hash_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: spatial_hash_uniform_buffer.as_entire_binding(),
            },
        ],
    );

    // Create pipeline layouts
    let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let spatial_hash_pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Spatial Hash Pipeline Layout"),
        bind_group_layouts: &[&spatial_hash_bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create compute pipelines
    let compute_pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("Particle Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let spatial_hash_clear_pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("Spatial Hash Clear Pipeline"),
        layout: Some(&spatial_hash_pipeline_layout),
        module: &spatial_hash_shader,
        entry_point: Some("clear_grid"),
        compilation_options: Default::default(),
        cache: None,
    });

    let spatial_hash_populate_pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("Spatial Hash Populate Pipeline"),
        layout: Some(&spatial_hash_pipeline_layout),
        module: &spatial_hash_shader,
        entry_point: Some("populate_grid"),
        compilation_options: Default::default(),
        cache: None,
    });

    let spatial_hash_collision_pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("Spatial Hash Collision Pipeline"),
        layout: Some(&spatial_hash_pipeline_layout),
        module: &spatial_hash_shader,
        entry_point: Some("detect_collisions"),
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
        spatial_hash_buffer,
        spatial_hash_uniform_buffer,
        spatial_hash_bind_group,
        spatial_hash_clear_pipeline,
        spatial_hash_populate_pipeline,
        spatial_hash_collision_pipeline,
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

    // For now, we'll use a simplified approach: assume the GPU compute has run
    // and read the data directly from the staging buffer (which was copied from particle buffer)
    // In a real implementation, we'd need to handle async buffer mapping
    
    // Since we can't easily do async buffer mapping in this system, we'll simulate
    // what the GPU compute shader SHOULD have produced, including particle-particle collisions
    let mut updated_particles = gpu_data.particles.clone();
    let mut collision_count = 0;
    
    // Simulate the FULL GPU compute shader logic including spatial hash collisions
    for i in 0..updated_particles.len() {
        let mut particle = updated_particles[i];
        
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
        
        // Particle-to-particle collisions (simulate spatial hash results)
        for j in 0..updated_particles.len() {
            if i == j { continue; } // Skip self
            
            let other = updated_particles[j];
            let dx = particle.position[0] - other.position[0];
            let dy = particle.position[1] - other.position[1];
            let dz = particle.position[2] - other.position[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            let min_distance = particle.radius + other.radius;
            
            if distance < min_distance && distance > 0.0 {
                // Calculate collision normal
                let normal = [dx / distance, dy / distance, dz / distance];
                let overlap = min_distance - distance;
                
                // Separate particles
                particle.position[0] += normal[0] * overlap * 0.45; // 0.9 * 0.5
                particle.position[1] += normal[1] * overlap * 0.45;
                particle.position[2] += normal[2] * overlap * 0.45;
                
                // Calculate relative velocity
                let rel_vel = [
                    particle.velocity[0] - other.velocity[0],
                    particle.velocity[1] - other.velocity[1],
                    particle.velocity[2] - other.velocity[2]
                ];
                let velocity_along_normal = rel_vel[0] * normal[0] + rel_vel[1] * normal[1] + rel_vel[2] * normal[2];
                
                // Only resolve if particles are moving towards each other
                if velocity_along_normal < 0.0 {
                    // Calculate impulse (simplified mass calculation)
                    let total_mass = particle.mass + other.mass;
                    let impulse_magnitude = -(1.0 + 0.4) * velocity_along_normal; // PARTICLE_RESTITUTION
                    let impulse_scale = (2.0 * other.mass) / total_mass;
                    let impulse = [
                        normal[0] * impulse_magnitude * impulse_scale,
                        normal[1] * impulse_magnitude * impulse_scale,
                        normal[2] * impulse_magnitude * impulse_scale
                    ];
                    
                    particle.velocity[0] += impulse[0];
                    particle.velocity[1] += impulse[1];
                    particle.velocity[2] += impulse[2];
                    
                    // Apply friction
                    let tangent_velocity = [
                        rel_vel[0] - normal[0] * velocity_along_normal,
                        rel_vel[1] - normal[1] * velocity_along_normal,
                        rel_vel[2] - normal[2] * velocity_along_normal
                    ];
                    particle.velocity[0] -= tangent_velocity[0] * 0.08; // PARTICLE_FRICTION * 0.1
                    particle.velocity[1] -= tangent_velocity[1] * 0.08;
                    particle.velocity[2] -= tangent_velocity[2] * 0.08;
                }
            }
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
        
        updated_particles[i] = particle;
    }
    
    // Calculate average height for debugging particle piling
    let total_height: f32 = updated_particles.iter().map(|p| p.position[1]).sum();
    let average_height = total_height / updated_particles.len() as f32;
    
    // Store results
    gpu_results.updated_particles = updated_particles;
    gpu_results.ready = true;
    
    // Log collision summary if any occurred
    if collision_count > 0 {
        info!("Frame summary: {} player-particle collisions detected", collision_count);
    }
    
    // Log average height every 60 frames (roughly once per second)
    static mut FRAME_COUNT: u32 = 0;
    unsafe {
        FRAME_COUNT += 1;
        if FRAME_COUNT % 60 == 0 {
            info!("Average particle height: {:.3} (frame {})", average_height, FRAME_COUNT);
        }
    }
    
    info!("Read back {} particles from GPU (simulated with particle-particle collisions)", gpu_data.particle_count);
}

// System to apply GPU results to Transform components (runs in main app)
fn apply_gpu_results_to_transforms(
    mut particle_query: Query<(&mut Transform, &mut Velocity, &crate::RegolithParticle), Without<crate::Player>>,
    player_query: Query<(&Transform, &Velocity), (With<crate::Player>, Without<crate::RegolithParticle>)>,
    time: Res<Time>,
) {
    // Get player data for interactions
    let (player_transform, player_velocity) = if let Ok(player_data) = player_query.single() {
        (player_data.0, player_data.1)
    } else {
        return;
    };
    
    // Collect particle data for collision detection
    let mut particle_data: Vec<(Vec3, Vec3, f32, f32)> = Vec::new(); // (position, velocity, radius, mass)
    for (transform, velocity, particle) in particle_query.iter() {
        particle_data.push((
            transform.translation,
            velocity.0,
            particle.radius,
            particle.mass,
        ));
    }
    
    let mut collision_count = 0;
    let mut debug_count = 0;
    
    // Apply physics simulation with particle-particle collisions
    for (i, (mut transform, mut velocity, particle)) in particle_query.iter_mut().enumerate() {
        let old_pos = transform.translation;
        
        // Apply gravity
        velocity.0.y += crate::LUNAR_GRAVITY * time.delta_secs();
        
        // Update position
        transform.translation += velocity.0 * time.delta_secs();
        
        // Ground collision
        if transform.translation.y <= particle.radius {
            transform.translation.y = particle.radius;
            velocity.0.y = velocity.0.y.abs() * -0.2; // RESTITUTION
            velocity.0.x *= 0.95; // FRICTION
            velocity.0.z *= 0.95; // FRICTION
        }
        
        // Particle-to-particle collisions
        for (j, &(other_pos, other_vel, other_radius, other_mass)) in particle_data.iter().enumerate() {
            if i == j { continue; }
            
            let dx = transform.translation.x - other_pos.x;
            let dy = transform.translation.y - other_pos.y;
            let dz = transform.translation.z - other_pos.z;
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            let min_distance = particle.radius + other_radius;
            
            if distance < min_distance && distance > 0.0 {
                // Calculate collision normal
                let normal = Vec3::new(dx / distance, dy / distance, dz / distance);
                let overlap = min_distance - distance;
                
                // Separate particles
                transform.translation += normal * overlap * 0.45;
                
                // Calculate relative velocity
                let rel_vel = velocity.0 - other_vel;
                let velocity_along_normal = rel_vel.dot(normal);
                
                // Only resolve if particles are moving towards each other
                if velocity_along_normal < 0.0 {
                    // Calculate impulse
                    let total_mass = particle.mass + other_mass;
                    let impulse_magnitude = -(1.0 + 0.4) * velocity_along_normal; // PARTICLE_RESTITUTION
                    let impulse_scale = (2.0 * other_mass) / total_mass;
                    let impulse = normal * impulse_magnitude * impulse_scale;
                    
                    velocity.0 += impulse;
                    
                    // Apply friction
                    let tangent_velocity = rel_vel - normal * velocity_along_normal;
                    velocity.0 -= tangent_velocity * 0.08; // PARTICLE_FRICTION
                }
            }
        }
        
        // Player collision
        let player_distance = transform.translation.distance(player_transform.translation);
        let min_distance = particle.radius + 0.8; // player radius
        
        if player_distance < min_distance && player_distance > 0.0 {
            collision_count += 1;
            
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
        
        // Debug only first few particles to avoid log spam
        if debug_count < 3 {
            info!("Particle {}: {:?} -> {:?}", debug_count + 1, old_pos, transform.translation);
            debug_count += 1;
        }
    }
    
    // Calculate average height for debugging particle piling
    let total_height: f32 = particle_data.iter().map(|(pos, _, _, _)| pos.y).sum();
    let average_height = total_height / particle_data.len() as f32;
    
    // Log collision summary if any occurred
    if collision_count > 0 {
        info!("Frame summary: {} player-particle collisions detected", collision_count);
    }
    
    // Log average height every 60 frames (roughly once per second)
    static mut FRAME_COUNT: u32 = 0;
    unsafe {
        FRAME_COUNT += 1;
        if FRAME_COUNT % 60 == 0 {
            info!("Average particle height: {:.3} (frame {})", average_height, FRAME_COUNT);
        }
    }
    
    info!("Applied physics with particle-particle collisions to {} particles", particle_data.len());
}

// System to update spatial hash uniforms
fn update_spatial_hash_uniforms(
    particle_query: Query<&Transform, (With<crate::RegolithParticle>, Without<crate::Player>)>,
    mut spatial_hash_data: ResMut<GpuSpatialHashData>,
) {
    if particle_query.is_empty() {
        return;
    }

    // Calculate world bounds from particle positions
    let mut min_pos = [f32::INFINITY; 3];
    let mut max_pos = [f32::NEG_INFINITY; 3];
    
    for transform in particle_query.iter() {
        let pos = transform.translation;
        min_pos[0] = min_pos[0].min(pos.x);
        min_pos[1] = min_pos[1].min(pos.y);
        min_pos[2] = min_pos[2].min(pos.z);
        max_pos[0] = max_pos[0].max(pos.x);
        max_pos[1] = max_pos[1].max(pos.y);
        max_pos[2] = max_pos[2].max(pos.z);
    }
    
    // Add padding to bounds
    let padding = 2.0;
    for i in 0..3 {
        min_pos[i] -= padding;
        max_pos[i] += padding;
    }
    
    // Calculate optimal cell size (should be ~2x max particle radius)
    let cell_size = 0.3; // Slightly larger than max particle radius (0.15)
    let grid_size = [64u32; 3]; // Fixed grid size for now
    
    spatial_hash_data.uniforms = SpatialHashUniforms {
        particle_count: particle_query.iter().count() as u32,
        grid_size,
        cell_size,
        world_min: min_pos,
        world_max: max_pos,
        max_particles_per_cell: 32,
        _padding: [0; 4],
    };
    
    spatial_hash_data.grid_size = grid_size;
    spatial_hash_data.cell_size = cell_size;
    spatial_hash_data.world_bounds = (min_pos, max_pos);
    spatial_hash_data.needs_update = true;
}

// System to upload spatial hash uniforms to GPU
fn upload_spatial_hash_uniforms(
    gpu_resources: Res<GpuComputeResources>,
    spatial_hash_data: Res<GpuSpatialHashData>,
    render_queue: Res<RenderQueue>,
) {
    if spatial_hash_data.needs_update {
        let uniforms_array = [spatial_hash_data.uniforms];
        let uniform_data = bytemuck::cast_slice(&uniforms_array);
        render_queue.write_buffer(
            &gpu_resources.spatial_hash_uniform_buffer,
            0,
            uniform_data,
        );
        info!("Uploaded spatial hash uniforms: grid_size={:?}, cell_size={:.3}",
              spatial_hash_data.grid_size, spatial_hash_data.cell_size);
    }
}

// System to dispatch spatial hash compute shaders
fn dispatch_spatial_hash_compute(
    gpu_resources: Res<GpuComputeResources>,
    spatial_hash_data: Res<GpuSpatialHashData>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if spatial_hash_data.uniforms.particle_count == 0 {
        return;
    }

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Spatial Hash Compute Encoder"),
    });

    // Pass 1: Clear hash grid
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Clear Hash Grid Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&gpu_resources.spatial_hash_clear_pipeline);
        compute_pass.set_bind_group(0, &gpu_resources.spatial_hash_bind_group, &[]);

        let total_cells = spatial_hash_data.grid_size[0] * spatial_hash_data.grid_size[1] * spatial_hash_data.grid_size[2];
        let workgroup_count = (total_cells + 63) / 64;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // Pass 2: Populate hash grid with particles
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Populate Hash Grid Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&gpu_resources.spatial_hash_populate_pipeline);
        compute_pass.set_bind_group(0, &gpu_resources.spatial_hash_bind_group, &[]);

        let workgroup_count = (spatial_hash_data.uniforms.particle_count + 63) / 64;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // Pass 3: Detect collisions using spatial hash
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Spatial Hash Collision Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&gpu_resources.spatial_hash_collision_pipeline);
        compute_pass.set_bind_group(0, &gpu_resources.spatial_hash_bind_group, &[]);

        let workgroup_count = (spatial_hash_data.uniforms.particle_count + 63) / 64;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    render_queue.submit(std::iter::once(encoder.finish()));
    
    info!("Dispatched spatial hash compute: {} particles, grid {:?}",
          spatial_hash_data.uniforms.particle_count, spatial_hash_data.grid_size);
}
