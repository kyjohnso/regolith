use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_rapier3d::prelude::*;
use clap::Parser;
use std::collections::VecDeque;

// Lunar gravity constant (1/6th of Earth's gravity)
const LUNAR_GRAVITY: f32 = -2.6; // m/s²
const PLAYER_SPEED: f32 = 2.0;
const JUMP_IMPULSE: f32 = 140.0; // Scaled for 70kg mass (2.0 * 70)
const PLAYER_MASS: f32 = 70.0; // kg - typical human mass

// Particle system constants
const PARTICLE_COUNT: usize = 15000;
const MIN_PARTICLE_RADIUS: f32 = 0.05;
const MAX_PARTICLE_RADIUS: f32 = 0.35;
const SPAWN_AREA_SIZE: f32 = 24.0;

// Command line arguments
#[derive(Parser, Debug)]
#[command(name = "regolith")]
#[command(about = "A lunar regolith simulation with randomized terrain")]
struct Args {
    /// Seed for terrain generation (default: 42)
    #[arg(short, long, default_value_t = 42)]
    seed: u64,
}

// Resource to store the terrain seed
#[derive(Resource)]
struct TerrainSeed(u64);

#[derive(Component)]
struct Player;

#[derive(Component)]
struct Truck;

#[derive(Component)]
struct TruckWheel {
    is_front: bool,
    is_left: bool,
}

#[derive(Component)]
struct WheelRotation {
    angular_velocity: f32,
}

#[derive(Component)]
struct RegolithParticle {
    radius: f32,
    mass: f32,
}

// FPS tracking resource
#[derive(Resource)]
struct FpsTracker {
    frame_times: VecDeque<f32>,
    current_fps: f32,
    update_timer: f32,
}

impl Default for FpsTracker {
    fn default() -> Self {
        Self {
            frame_times: VecDeque::with_capacity(60),
            current_fps: 0.0,
            update_timer: 0.0,
        }
    }
}

// Component for FPS display text
#[derive(Component)]
struct FpsText;

fn main() {
    let args = Args::parse();
    
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default())
        .init_resource::<FpsTracker>()
        .insert_resource(TerrainSeed(args.seed))
        .add_systems(Startup, (setup, spawn_regolith_particles, setup_fps_ui))
        .add_systems(Update, (
            player_input,
            player_movement,
            wheel_rotation_system,
            fps_tracker_system,
            fps_display_system,
        ))
        .run();
}

// Function to create a hilly terrain mesh with randomized parameters
fn create_hilly_terrain(size: f32, resolution: usize, seed: u64) -> Mesh {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    
    let step = size / resolution as f32;
    let half_size = size * 0.5;
    
    // Create seeded random number generator for reproducible terrain
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Generate random coefficients for terrain layers
    let num_layers = rng.gen_range(4..8); // Random number of terrain layers
    let mut terrain_layers = Vec::new();
    
    for _ in 0..num_layers {
        terrain_layers.push((
            rng.gen_range(0.01..0.3),   // frequency_x
            rng.gen_range(0.01..0.3),   // frequency_z
            rng.gen_range(0.5..4.0),    // amplitude
            rng.gen_range(0.0..std::f32::consts::TAU), // phase_x
            rng.gen_range(0.0..std::f32::consts::TAU), // phase_z
            rng.gen_range(0..4),        // wave_type (0=sin*cos, 1=cos*sin, 2=sin*sin, 3=cos*cos)
        ));
    }
    
    // Generate height map using randomized multi-layer noise
    let mut heights = vec![vec![0.0; resolution + 1]; resolution + 1];
    for x in 0..=resolution {
        for z in 0..=resolution {
            let world_x = (x as f32 * step) - half_size;
            let world_z = (z as f32 * step) - half_size;
            
            let mut height = 0.0;
            
            // Apply each terrain layer with random parameters
            for &(freq_x, freq_z, amplitude, phase_x, phase_z, wave_type) in &terrain_layers {
                let wave_x = world_x * freq_x + phase_x;
                let wave_z = world_z * freq_z + phase_z;
                
                let layer_height = match wave_type {
                    0 => wave_x.sin() * wave_z.cos(),
                    1 => wave_x.cos() * wave_z.sin(),
                    2 => wave_x.sin() * wave_z.sin(),
                    3 => wave_x.cos() * wave_z.cos(),
                    _ => wave_x.sin() * wave_z.cos(),
                } * amplitude;
                
                height += layer_height;
            }
            
            // Add some high-frequency detail noise
            let detail_noise =
                (world_x * rng.gen_range(0.3..0.8) + rng.gen_range(0.0..std::f32::consts::TAU)).sin() *
                (world_z * rng.gen_range(0.3..0.8) + rng.gen_range(0.0..std::f32::consts::TAU)).cos() *
                rng.gen_range(0.1..0.5);
            
            height += detail_noise;
            heights[x][z] = height;
        }
    }
    
    // Generate vertices
    for x in 0..=resolution {
        for z in 0..=resolution {
            let world_x = (x as f32 * step) - half_size;
            let world_z = (z as f32 * step) - half_size;
            let height = heights[x][z];
            
            positions.push([world_x, height, world_z]);
            uvs.push([x as f32 / resolution as f32, z as f32 / resolution as f32]);
            
            // Calculate normal using neighboring heights
            let normal = if x > 0 && x < resolution && z > 0 && z < resolution {
                let left = heights[x - 1][z];
                let right = heights[x + 1][z];
                let up = heights[x][z - 1];
                let down = heights[x][z + 1];
                
                let dx = right - left;
                let dz = down - up;
                
                Vec3::new(-dx, 2.0 * step, -dz).normalize()
            } else {
                Vec3::Y // Default normal for edge vertices
            };
            
            normals.push([normal.x, normal.y, normal.z]);
        }
    }
    
    // Generate indices for triangles
    for x in 0..resolution {
        for z in 0..resolution {
            let i = x * (resolution + 1) + z;
            let i_next_row = (x + 1) * (resolution + 1) + z;
            
            // First triangle
            indices.push(i as u32);
            indices.push((i + 1) as u32);
            indices.push(i_next_row as u32);
            
            // Second triangle
            indices.push((i + 1) as u32);
            indices.push((i_next_row + 1) as u32);
            indices.push(i_next_row as u32);
        }
    }
    
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    
    mesh
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    terrain_seed: Res<TerrainSeed>,
) {
    // Create terrain mesh for collision using the command line seed
    let seed = terrain_seed.0;
    let terrain_mesh = create_hilly_terrain(100.0, 100, seed);
    let terrain_mesh_handle = meshes.add(terrain_mesh.clone());
    
    println!("Generated terrain with seed: {} (use --seed <number> to change)", seed);
    
    // Add truck with Rapier rigid body and cuboid collider
    let truck_entity = commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(3.0, 1.0, 1.5))), // Truck body: length x height x width
        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.2, 0.2))), // Red truck
        Transform::from_xyz(0.0, 5.0, 0.0),
        Player,
        Truck,
        RigidBody::Dynamic,
        Collider::cuboid(1.5, 0.5, 0.75), // Half extents for truck body
        Restitution::coefficient(0.2),
        Friction::coefficient(0.8),
        Damping { linear_damping: 0.2, angular_damping: 1.0 },
        // Allow full rotation so truck can tilt on terrain
        AdditionalMassProperties::Mass(PLAYER_MASS * 2.0), // Heavier truck
        ExternalForce::default(),
        ExternalImpulse::default(),
    )).id();

    // Add wheels as visual children of the truck (not separate physics bodies)
    let wheel_configs = [
        (Vec3::new(1.2, -0.7, 0.9), true, false),   // Front right
        (Vec3::new(1.2, -0.7, -0.9), true, true),   // Front left
        (Vec3::new(-1.2, -0.7, 0.9), false, false), // Rear right
        (Vec3::new(-1.2, -0.7, -0.9), false, true), // Rear left
    ];

    for (wheel_pos, is_front, is_left) in wheel_configs {
        let wheel_entity = commands.spawn((
            Mesh3d(meshes.add(Cylinder::new(0.4, 0.2))), // Wheel: radius, height
            MeshMaterial3d(materials.add(Color::srgb(0.1, 0.1, 0.1))), // Black wheels
            Transform::from_translation(wheel_pos)
                .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)), // Rotate to be wheel-like
            TruckWheel { is_front, is_left },
            WheelRotation { angular_velocity: 0.0 },
        )).id();
        
        // Make wheels children of the truck so they move together
        commands.entity(truck_entity).add_child(wheel_entity);
    }

    // Add a basic cube to verify the scene is working
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.0, 2.0, 2.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.7, 0.6))),
        Transform::from_xyz(5.0, 3.0, 0.0),
        RigidBody::Dynamic,
        Collider::cuboid(1.0, 1.0, 1.0),
        Restitution::coefficient(0.4),
        Friction::coefficient(0.6),
    ));

    // Add lunar surface terrain with Rapier collider
    commands.spawn((
        Mesh3d(terrain_mesh_handle.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.4, 0.4, 0.35), // Lunar regolith gray-brown
            perceptual_roughness: 0.9, // Very rough surface like moon dust
            metallic: 0.0,  // Non-metallic
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        RigidBody::Fixed,
        Collider::from_bevy_mesh(&terrain_mesh, &ComputedColliderShape::TriMesh(TriMeshFlags::empty())).unwrap(),
        Friction::coefficient(0.8),
        Restitution::coefficient(0.1),
    ));

    // Add a light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::ZYX, 0.0, 1.0, -std::f32::consts::FRAC_PI_4)),
    ));

    // Add orbit camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-4.0, 4.5, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}

fn player_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut truck_query: Query<(&mut ExternalForce, &mut ExternalImpulse, &Transform), (With<Player>, With<Truck>)>,
    mut wheel_query: Query<&mut WheelRotation, With<TruckWheel>>,
    _time: Res<Time>,
) {
    let mut drive_input = 0.0;
    let mut steer_input = 0.0;

    // Get input
    if keyboard_input.pressed(KeyCode::KeyW) {
        drive_input = 1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyS) {
        drive_input = -1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyA) {
        steer_input = -1.0;
    }
    if keyboard_input.pressed(KeyCode::KeyD) {
        steer_input = 1.0;
    }

    // Update wheel rotation for visual effect
    for mut wheel_rotation in &mut wheel_query {
        wheel_rotation.angular_velocity = drive_input * 8.0;
    }

    // Apply forces to truck body for movement
    for (mut external_force, mut external_impulse, transform) in &mut truck_query {
        let mut movement = Vec3::ZERO;

        // Forward/backward movement
        if drive_input != 0.0 {
            let forward_dir = transform.forward();
            movement += forward_dir * drive_input * PLAYER_SPEED * PLAYER_MASS * 50.0;
        }

        // Turning - apply torque around Y axis
        if steer_input != 0.0 {
            let turn_torque = Vec3::new(0.0, steer_input * PLAYER_MASS * 25.0, 0.0);
            external_force.torque = turn_torque;
        } else {
            external_force.torque = Vec3::ZERO;
        }

        external_force.force = movement;

        // Jump (Space key)
        if keyboard_input.just_pressed(KeyCode::Space) {
            external_impulse.impulse = Vec3::new(0.0, JUMP_IMPULSE * 20.0, 0.0);
        }
    }
}

fn player_movement(
    mut player_query: Query<&mut Transform, With<Player>>,
) {
    // Rapier handles all physics movement automatically
    // This system can be used for additional player-specific logic if needed
    for _transform in &mut player_query {
        // Player movement is now handled by Rapier physics
        // Any additional player logic can go here
    }
}

fn wheel_rotation_system(
    mut wheel_query: Query<(&mut Transform, &WheelRotation), With<TruckWheel>>,
    time: Res<Time>,
) {
    for (mut transform, wheel_rotation) in &mut wheel_query {
        // Rotate the wheel mesh based on angular velocity
        // Since wheels are cylinders rotated 90 degrees, we rotate around local Y axis for rolling motion
        let rotation_delta = wheel_rotation.angular_velocity * time.delta_secs();
        transform.rotate_local_y(rotation_delta);
    }
}

fn spawn_regolith_particles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Create materials for different particle types
    let fine_dust_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.5, 0.4, 0.3), // Darker for fine dust
        perceptual_roughness: 0.9,
        metallic: 0.0,
        ..default()
    });
    
    let medium_particle_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.6, 0.5, 0.4), // Standard regolith brown-gray
        perceptual_roughness: 0.8,
        metallic: 0.0,
        ..default()
    });
    
    let large_particle_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.6, 0.5), // Lighter for larger particles
        perceptual_roughness: 0.7,
        metallic: 0.1, // Slight metallic for rock-like appearance
        ..default()
    });

    // Spawn particles with varied sizes using Rapier physics
    for _ in 0..PARTICLE_COUNT {
        let x = rng.gen_range(-SPAWN_AREA_SIZE..SPAWN_AREA_SIZE);
        let z = rng.gen_range(-SPAWN_AREA_SIZE..SPAWN_AREA_SIZE);
        let y = rng.gen_range(3.0..8.0); // Start particles higher to let them fall

        // Generate random particle radius
        let radius = rng.gen_range(MIN_PARTICLE_RADIUS..MAX_PARTICLE_RADIUS);
        
        // Mass scales with volume (radius^3) for realistic physics
        let mass = (radius / MIN_PARTICLE_RADIUS).powi(3) * 0.5;
        
        // Create individual mesh for each particle size
        let particle_mesh = meshes.add(Sphere::new(radius));
        
        // Choose material based on particle size
        let material = if radius < MIN_PARTICLE_RADIUS + 0.03 {
            fine_dust_material.clone()
        } else if radius < MIN_PARTICLE_RADIUS + 0.07 {
            medium_particle_material.clone()
        } else {
            large_particle_material.clone()
        };

        commands.spawn((
            Mesh3d(particle_mesh),
            MeshMaterial3d(material),
            Transform::from_xyz(x, y, z),
            RegolithParticle { radius, mass },
            RigidBody::Dynamic,
            Collider::ball(radius), // Simple sphere collider for particles
            Restitution::coefficient(0.2), // Low bounce for regolith
            Friction::coefficient(0.8), // High friction for dust-like behavior
            Damping { linear_damping: 0.3, angular_damping: 0.5 }, // Damping to simulate dust behavior
            AdditionalMassProperties::Mass(mass),
            Sleeping::disabled(), // Allow particles to sleep when stable
        ));
    }

    println!("Spawned {} regolith particles with Rapier physics ({:.3} to {:.3})",
             PARTICLE_COUNT, MIN_PARTICLE_RADIUS, MAX_PARTICLE_RADIUS);
}

// FPS tracking system
fn fps_tracker_system(
    mut fps_tracker: ResMut<FpsTracker>,
    time: Res<Time>,
) {
    let delta_time = time.delta_secs();
    
    // Add current frame time to the queue
    fps_tracker.frame_times.push_back(delta_time);
    
    // Keep only the last 60 frames (1 second at 60 FPS)
    if fps_tracker.frame_times.len() > 60 {
        fps_tracker.frame_times.pop_front();
    }
    
    // Update FPS calculation every 0.1 seconds
    fps_tracker.update_timer += delta_time;
    if fps_tracker.update_timer >= 0.1 {
        fps_tracker.update_timer = 0.0;
        
        // Calculate average frame time and convert to FPS
        if !fps_tracker.frame_times.is_empty() {
            let avg_frame_time: f32 = fps_tracker.frame_times.iter().sum::<f32>() / fps_tracker.frame_times.len() as f32;
            fps_tracker.current_fps = if avg_frame_time > 0.0 { 1.0 / avg_frame_time } else { 0.0 };
        }
    }
}

// Setup FPS UI overlay
fn setup_fps_ui(mut commands: Commands) {
    // Create UI camera with higher priority to render on top
    commands.spawn((
        Camera2d,
        Camera {
            order: 1, // Higher priority than the 3D camera (which defaults to 0)
            ..default()
        },
    ));
    
    // Create FPS text overlay
    commands.spawn((
        Text::new("FPS: 0"),
        TextFont {
            font_size: 24.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 1.0, 0.0)), // Yellow text
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        FpsText,
    ));
}

// Update FPS display system
fn fps_display_system(
    fps_tracker: Res<FpsTracker>,
    mut fps_text_query: Query<&mut Text, With<FpsText>>,
) {
    if let Ok(mut text) = fps_text_query.single_mut() {
        text.0 = format!("FPS: {:.1}", fps_tracker.current_fps);
    }
}
