use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

mod gpu_compute;
use gpu_compute::{GpuComputePlugin, ComputeUniforms};

// Lunar gravity constant (1/6th of Earth's gravity)
const LUNAR_GRAVITY: f32 = -1.62; // m/s²
const PLAYER_SPEED: f32 =2.0;
const JUMP_IMPULSE: f32 = 2.0;

// Particle system constants
const PARTICLE_COUNT: usize = 1000; // Reduce for better visibility of interactions
const PARTICLE_RADIUS: f32 = 0.1; // Make particles bigger for easier visibility
const SPAWN_AREA_SIZE: f32 = 2.0; // Spawn even closer to player for testing

// GPU compute toggle
const USE_GPU_COMPUTE: bool = false; // Set to false to use CPU physics - testing first

#[derive(Component)]
struct Player;

#[derive(Component)]
struct Velocity(Vec3);

#[derive(Component)]
struct Grounded(bool);

#[derive(Component)]
struct RegolithParticle {
    radius: f32,
    mass: f32,
}

fn main() {
    let mut app = App::new();
    
    app.add_plugins(DefaultPlugins)
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, (setup, spawn_regolith_particles));
    
    if USE_GPU_COMPUTE {
        println!("Using GPU compute for particle physics");
        app.add_plugins(GpuComputePlugin)
            .add_systems(Update, (
                player_movement,
                player_input,
                particle_physics,
            ));
    } else {
        println!("Using CPU for particle physics");
        app.add_systems(Update, (
            player_movement,
            apply_gravity,
            player_input,
            particle_physics,
            particle_collisions,
            player_particle_interactions,
        ));
    }
    
    app.run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Add player (represented as a capsule for now)
    commands.spawn((
        Mesh3d(meshes.add(Capsule3d::new(0.5, 1.8))),
        MeshMaterial3d(materials.add(Color::srgb(0.2, 0.7, 0.9))),
        Transform::from_xyz(0.0, 2.0, 0.0),
        Player,
        Velocity(Vec3::ZERO),
        Grounded(false),
    ));

    // Add a basic cube to verify the scene is working
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.0, 2.0, 2.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.7, 0.6))),
        Transform::from_xyz(5.0, 1.0, 0.0),
    ));

    // Add lunar surface terrain - larger plane with lunar-like coloring
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(100.0, 100.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.4, 0.4, 0.35), // Lunar regolith gray-brown
            perceptual_roughness: 0.9, // Very rough surface like moon dust
            metallic: 0.0,  // Non-metallic
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
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
    mut player_query: Query<(&mut Velocity, &Grounded), With<Player>>,
    _time: Res<Time>,
) {
    for (mut velocity, grounded) in &mut player_query {
        let mut movement = Vec3::ZERO;

        // Horizontal movement (WASD)
        if keyboard_input.pressed(KeyCode::KeyW) {
            movement.z -= 1.0;
        }
        if keyboard_input.pressed(KeyCode::KeyS) {
            movement.z += 1.0;
        }
        if keyboard_input.pressed(KeyCode::KeyA) {
            movement.x -= 1.0;
        }
        if keyboard_input.pressed(KeyCode::KeyD) {
            movement.x += 1.0;
        }

        // Normalize and apply speed
        if movement.length() > 0.0 {
            movement = movement.normalize() * PLAYER_SPEED;
            velocity.0.x = movement.x;
            velocity.0.z = movement.z;
        } else {
            // Apply friction when not moving
            velocity.0.x *= 0.8;
            velocity.0.z *= 0.8;
        }

        // Jump (Space key)
        if keyboard_input.just_pressed(KeyCode::Space) && grounded.0 {
            velocity.0.y = JUMP_IMPULSE;
        }
    }
}

fn apply_gravity(
    mut player_query: Query<&mut Velocity, With<Player>>,
    time: Res<Time>,
) {
    for mut velocity in &mut player_query {
        // Apply lunar gravity
        velocity.0.y += LUNAR_GRAVITY * time.delta_secs();
    }
}

fn player_movement(
    mut player_query: Query<(&mut Transform, &mut Velocity, &mut Grounded), With<Player>>,
    time: Res<Time>,
) {
    for (mut transform, mut velocity, mut grounded) in &mut player_query {
        // Apply velocity to position
        transform.translation += velocity.0 * time.delta_secs();

        // Simple ground collision (y = 0.9 is half the capsule height)
        if transform.translation.y <= 0.9 {
            transform.translation.y = 0.9;
            velocity.0.y = 0.0;
            grounded.0 = true;
        } else {
            grounded.0 = false;
        }
    }
}

fn spawn_regolith_particles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Create a shared mesh and material for all particles
    let particle_mesh = meshes.add(Sphere::new(PARTICLE_RADIUS));
    let particle_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.6, 0.5, 0.4), // Regolith brown-gray
        perceptual_roughness: 0.8,
        metallic: 0.0,
        ..default()
    });

    // Spawn particles in a random distribution above the surface, closer to player
    for _ in 0..PARTICLE_COUNT {
        let x = rng.gen_range(-SPAWN_AREA_SIZE..SPAWN_AREA_SIZE);
        let z = rng.gen_range(-SPAWN_AREA_SIZE..SPAWN_AREA_SIZE);
        let y = rng.gen_range(1.0..5.0); // Start particles closer to ground and player

        commands.spawn((
            Mesh3d(particle_mesh.clone()),
            MeshMaterial3d(particle_material.clone()),
            Transform::from_xyz(x, y, z),
            RegolithParticle {
                radius: PARTICLE_RADIUS,
                mass: 1.0, // Simple uniform mass for now
            },
            Velocity(Vec3::ZERO),
        ));
    }

    println!("Spawned {} regolith particles", PARTICLE_COUNT);
}

fn particle_collisions(
    mut particle_query: Query<(&mut Transform, &mut Velocity, &RegolithParticle), Without<Player>>,
) {
    // Collect all particle data first
    let mut particle_data: Vec<(Vec3, Vec3, f32)> = Vec::new();
    
    for (transform, velocity, particle) in particle_query.iter() {
        particle_data.push((transform.translation, velocity.0, particle.radius));
    }
    
    // Calculate collision responses
    let mut velocity_updates: Vec<Vec3> = vec![Vec3::ZERO; particle_data.len()];
    let mut position_updates: Vec<Vec3> = vec![Vec3::ZERO; particle_data.len()];
    
    // Simple O(n²) collision detection - we'll optimize this later with spatial hashing
    for i in 0..particle_data.len() {
        for j in (i + 1)..particle_data.len() {
            let (pos_a, vel_a, radius_a) = particle_data[i];
            let (pos_b, vel_b, radius_b) = particle_data[j];
            
            let distance = pos_a.distance(pos_b);
            let min_distance = radius_a + radius_b;
            
            // Check for collision
            if distance < min_distance && distance > 0.0 {
                // Calculate collision normal
                let normal = (pos_b - pos_a).normalize();
                
                // Separate particles to prevent overlap
                let overlap = min_distance - distance;
                let separation = normal * (overlap * 0.5);
                
                position_updates[i] -= separation;
                position_updates[j] += separation;
                
                // Simple elastic collision response
                let relative_velocity = vel_b - vel_a;
                let velocity_along_normal = relative_velocity.dot(normal);
                
                // Don't resolve if velocities are separating
                if velocity_along_normal > 0.0 {
                    continue;
                }
                
                // Restitution (bounciness) - lower for regolith
                let restitution = 0.2;
                let impulse_scalar = -(1.0 + restitution) * velocity_along_normal;
                let impulse = normal * impulse_scalar;
                
                // Apply impulse (assuming equal mass for simplicity)
                velocity_updates[i] -= impulse * 0.5;
                velocity_updates[j] += impulse * 0.5;
            }
        }
    }
    
    // Apply updates to actual particles
    for (i, (mut transform, mut velocity, _)) in particle_query.iter_mut().enumerate() {
        transform.translation += position_updates[i];
        velocity.0 += velocity_updates[i];
        
        // Add some damping to simulate energy loss
        velocity.0 *= 0.98;
    }
}

fn player_particle_interactions(
    mut player_query: Query<(&mut Transform, &mut Velocity), With<Player>>,
    mut particle_query: Query<(&mut Transform, &mut Velocity, &RegolithParticle), Without<Player>>,
) {
    // Get player data
    if let Ok((mut player_transform, mut player_velocity)) = player_query.single_mut() {
        let player_radius = 0.8; // Increase player interaction radius to push more particles
        let mut collision_count = 0;
        
        // Check collisions with all particles
        for (mut particle_transform, mut particle_velocity, particle) in particle_query.iter_mut() {
            let distance = player_transform.translation.distance(particle_transform.translation);
            let min_distance = player_radius + particle.radius;
            
            // Check for collision or close proximity
            if distance < min_distance && distance > 0.0 {
                collision_count += 1;
                // Calculate collision normal (from player to particle)
                let normal = (particle_transform.translation - player_transform.translation).normalize();
                
                // Separate player and particle
                let overlap = min_distance - distance;
                let separation = normal * overlap;
                
                // Move particle away from player (player is much heavier)
                particle_transform.translation += separation * 0.9;
                player_transform.translation -= separation * 0.1;
                
                // Transfer player momentum to particle - make this more aggressive
                let player_speed = player_velocity.0.length();
                if player_speed > 0.05 {
                    // Player pushes particles in movement direction with more force
                    let push_force = player_velocity.0.normalize() * (player_speed * 1.5);
                    particle_velocity.0 += push_force;
                    
                    // Player loses some momentum when hitting particles
                    player_velocity.0 *= 0.98;
                } else {
                    // Even when player is stationary, push particles away
                    let push_direction = normal;
                    particle_velocity.0 += push_direction * 3.0; // Stronger minimum push force
                }
                
                // Add some bounce to the particle with more energy
                let relative_velocity = player_velocity.0 - particle_velocity.0;
                let velocity_along_normal = relative_velocity.dot(normal);
                
                if velocity_along_normal < 0.0 {
                    let restitution = 0.5; // More bouncy
                    let impulse_scalar = -(1.0 + restitution) * velocity_along_normal;
                    let impulse = normal * impulse_scalar;
                    
                    // Apply impulse (player is much heavier, so gets less effect)
                    particle_velocity.0 += impulse * 0.9;
                    player_velocity.0 -= impulse * 0.1;
                }
            }
        }
        
        // Debug output every few frames to see interaction count
        if collision_count > 0 {
            println!("Player interacting with {} particles", collision_count);
        }
    }
}

fn particle_physics(
    mut particle_query: Query<(&mut Transform, &mut Velocity, &RegolithParticle), Without<Player>>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();

    // Apply gravity to all particles
    for (mut transform, mut velocity, _particle) in &mut particle_query {
        // Apply lunar gravity
        velocity.0.y += LUNAR_GRAVITY * dt;

        // Apply velocity to position
        transform.translation += velocity.0 * dt;

        // Simple ground collision
        if transform.translation.y <= PARTICLE_RADIUS {
            transform.translation.y = PARTICLE_RADIUS;
            velocity.0.y = velocity.0.y.abs() * -0.2; // Bounce with energy loss
            velocity.0.x *= 0.95; // Reduced friction - particles slide more on lunar surface
            velocity.0.z *= 0.95; // Reduced friction - particles slide more on lunar surface
        }
        
        // GPU-based particle physics system
        fn gpu_particle_physics(
            player_query: Query<(&Transform, &Velocity), With<Player>>,
            particle_query: Query<&Transform, (With<RegolithParticle>, Without<Player>)>,
            time: Res<Time>,
        ) {
            // This system will interface with the GPU compute shader
            // For now, it's a placeholder that will be implemented in the next step
            
            if let Ok((player_transform, player_velocity)) = player_query.single() {
                let particle_count = particle_query.iter().count();
                
                // Prepare uniforms for GPU compute
                let _uniforms = ComputeUniforms {
                    delta_time: time.delta_secs(),
                    gravity: LUNAR_GRAVITY,
                    particle_count: particle_count as u32,
                    ground_level: PARTICLE_RADIUS,
                    player_position: [
                        player_transform.translation.x,
                        player_transform.translation.y,
                        player_transform.translation.z,
                    ],
                    player_radius: 0.5,
                    player_velocity: [
                        player_velocity.0.x,
                        player_velocity.0.y,
                        player_velocity.0.z,
                    ],
                    _padding: 0.0,
                };
                
                // TODO: Dispatch compute shader with uniforms
                // This will be implemented in the next step
            }
        }
    }
}