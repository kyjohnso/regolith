use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{RenderApp, Render};
use bytemuck::{Pod, Zeroable};

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
#[derive(Clone, Copy, Pod, Zeroable)]
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
    pub bind_group: BindGroup,
    pub compute_pipeline: ComputePipeline,
}

// Plugin for GPU compute
pub struct GpuComputePlugin;

impl Plugin for GpuComputePlugin {
    fn build(&self, app: &mut App) {
        // Add systems to the render app
        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            (
                setup_gpu_compute.run_if(not(resource_exists::<GpuComputeResources>)),
                dispatch_compute_shader.run_if(resource_exists::<GpuComputeResources>),
            ),
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
        bind_group,
        compute_pipeline,
    });

    info!("GPU compute resources initialized successfully!");
}

// System to dispatch the compute shader
fn dispatch_compute_shader(
    gpu_resources: Res<GpuComputeResources>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
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
        let workgroup_count = (1000 + 63) / 64; // Round up division for 1000 particles
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // Submit commands
    render_queue.submit(std::iter::once(encoder.finish()));
}