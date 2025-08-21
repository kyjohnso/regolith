use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use std::collections::VecDeque;

mod gpu_compute;
use gpu_compute::{GpuComputePlugin, ComputeUniforms};

// Lunar gravity constant (1/6th of Earth's gravity)
const LUNAR_GRAVITY: f32 = -1.62; // m/s²
const PLAYER_SPEED: f32 =2.0;
const JUMP_IMPULSE: f32 = 2.0;

// Particle system constants
const PARTICLE_COUNT: usize = 1000; // Reduce for better visibility of interactions
const MIN_PARTICLE_RADIUS: f32 = 0.05; // Smallest particles (fine dust)
const MAX_PARTICLE_RADIUS: f32 = 0.15; // Largest particles (small rocks)
const SPAWN_AREA_SIZE: f32 = 4.0; // Spawn even closer to player for testing

// Damping constants
const VELOCITY_DAMPING: f32 = 0.95; // General velocity damping (very aggressive to reduce jittering)
const COLLISION_DAMPING: f32 = 0.6; // Additional damping after collisions (very aggressive)
const MIN_VELOCITY_THRESHOLD: f32 = 0.08; // Velocities below this are set to zero (higher threshold)
const GROUND_DAMPING: f32 = 0.85; // Additional damping for particles near ground
const ANGULAR_DAMPING: f32 = 0.9; // For rotational motion if we add it later

// GPU compute toggle
const USE_GPU_COMPUTE: bool = false; // Switch back to CPU physics - GPU integration needs more work

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
    let mut app = App::new();
    
    app.add_plugins(DefaultPlugins)
        .add_plugins(PanOrbitCameraPlugin)
        .init_resource::<FpsTracker>()
        .add_systems(Startup, (setup, spawn_regolith_particles, setup_fps_ui))
        .add_systems(Update, (fps_tracker_system, fps_display_system));
    
    if USE_GPU_COMPUTE {
        println!("Using GPU compute for particle physics");
        app.add_plugins(GpuComputePlugin)
            .add_systems(Update, (
                player_movement,
                apply_gravity,
                player_input,
                gpu_particle_physics,
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
            monitor_particle_stability,
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

    // Spawn particles with varied sizes in a random distribution above the surface
    for _ in 0..PARTICLE_COUNT {
        let x = rng.gen_range(-SPAWN_AREA_SIZE..SPAWN_AREA_SIZE);
        let z = rng.gen_range(-SPAWN_AREA_SIZE..SPAWN_AREA_SIZE);
        let y = rng.gen_range(1.0..5.0); // Start particles closer to ground and player

        // Generate random particle radius
        let radius = rng.gen_range(MIN_PARTICLE_RADIUS..MAX_PARTICLE_RADIUS);
        
        // Mass scales with volume (radius^3) for realistic physics
        let mass = (radius / MIN_PARTICLE_RADIUS).powi(3) * 0.5; // Base mass of 0.5 for smallest particles
        
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
            RegolithParticle {
                radius,
                mass,
            },
            Velocity(Vec3::ZERO),
        ));
    }

    println!("Spawned {} regolith particles with varied sizes ({:.3} to {:.3})",
             PARTICLE_COUNT, MIN_PARTICLE_RADIUS, MAX_PARTICLE_RADIUS);
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
                
                // Get masses from particle data (we need to access the actual particles)
                // For now, we'll use a simplified approach based on radius
                let mass_a = (radius_a / MIN_PARTICLE_RADIUS).powi(3) * 0.5;
                let mass_b = (radius_b / MIN_PARTICLE_RADIUS).powi(3) * 0.5;
                let total_mass = mass_a + mass_b;
                
                // Mass-based elastic collision response
                let relative_velocity = vel_b - vel_a;
                let velocity_along_normal = relative_velocity.dot(normal);
                
                // Don't resolve if velocities are separating
                if velocity_along_normal > 0.0 {
                    continue;
                }
                
                // Restitution (bounciness) - lower for regolith
                let restitution = 0.2;
                let impulse_scalar = -(1.0 + restitution) * velocity_along_normal / total_mass;
                let impulse = normal * impulse_scalar;
                
                // Apply impulse based on mass ratios (heavier particles move less)
                velocity_updates[i] -= impulse * mass_b;
                velocity_updates[j] += impulse * mass_a;
            }
        }
    }
    
    // Apply updates to actual particles
    for (i, (mut transform, mut velocity, _)) in particle_query.iter_mut().enumerate() {
        transform.translation += position_updates[i];
        velocity.0 += velocity_updates[i];
        
        // Apply collision damping if there was a collision
        if velocity_updates[i].length() > 0.001 {
            velocity.0 *= COLLISION_DAMPING;
        }
        
        // Apply general velocity damping to simulate energy loss
        velocity.0 *= VELOCITY_DAMPING;
        
        // Clamp very small velocities to zero to prevent micro-jittering
        if velocity.0.length() < MIN_VELOCITY_THRESHOLD {
            velocity.0 = Vec3::ZERO;
        }
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
                
                // Calculate mass-based interaction (player is much heavier)
                let player_mass = 70.0; // kg - typical human mass
                let particle_mass = particle.mass;
                let mass_ratio = particle_mass / (player_mass + particle_mass);
                
                // Transfer player momentum to particle - scaled by mass difference
                let player_speed = player_velocity.0.length();
                if player_speed > 0.05 {
                    // Player pushes particles in movement direction, smaller particles get pushed more
                    let push_force = player_velocity.0.normalize() * (player_speed * 2.0 / particle_mass.sqrt());
                    particle_velocity.0 += push_force;
                    
                    // Player loses momentum proportional to particle mass
                    player_velocity.0 *= 1.0 - (mass_ratio * 0.05);
                } else {
                    // Even when player is stationary, push particles away (smaller particles move more)
                    let push_direction = normal;
                    particle_velocity.0 += push_direction * (4.0 / particle_mass.sqrt());
                }
                
                // Add mass-based bounce to the particle
                let relative_velocity = player_velocity.0 - particle_velocity.0;
                let velocity_along_normal = relative_velocity.dot(normal);
                
                if velocity_along_normal < 0.0 {
                    let restitution = 0.5; // More bouncy
                    let impulse_scalar = -(1.0 + restitution) * velocity_along_normal;
                    let impulse = normal * impulse_scalar;
                    
                    // Apply impulse based on mass ratios
                    particle_velocity.0 += impulse * (1.0 - mass_ratio);
                    player_velocity.0 -= impulse * mass_ratio;
                    
                    // Additional damping after player collision to reduce jittering
                    particle_velocity.0 *= COLLISION_DAMPING;
                }
                
                // Clamp small velocities to prevent micro-jittering after player interaction
                if particle_velocity.0.length() < MIN_VELOCITY_THRESHOLD {
                    particle_velocity.0 = Vec3::ZERO;
                }
            }
        }
        
        // // Debug output every few frames to see interaction count
        // if collision_count > 0 {
        //     println!("Player interacting with {} particles", collision_count);
        // }
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

        // Simple ground collision using individual particle radius
        if transform.translation.y <= _particle.radius {
            transform.translation.y = _particle.radius;
            velocity.0.y = velocity.0.y.abs() * -0.2; // Bounce with energy loss
            velocity.0.x *= 0.95; // Reduced friction - particles slide more on lunar surface
            velocity.0.z *= 0.95; // Reduced friction - particles slide more on lunar surface
            
            // Additional damping after ground collision to prevent bouncing jitter
            velocity.0 *= COLLISION_DAMPING;
        }
        
        // Apply general velocity damping
        velocity.0 *= VELOCITY_DAMPING;
        
        // Clamp very small velocities to zero to prevent micro-jittering
        if velocity.0.length() < MIN_VELOCITY_THRESHOLD {
            velocity.0 = Vec3::ZERO;
        }
        
        // Additional stability check: if particle is very close to ground and moving slowly
        if transform.translation.y <= _particle.radius + 0.02 {
            // Apply extra damping for particles near ground
            velocity.0 *= GROUND_DAMPING;
            
            // Stop vertical jittering near ground
            if velocity.0.y.abs() < 0.1 {
                velocity.0.y = 0.0;
            }
            
            // Stop horizontal micro-movements near ground
            if velocity.0.xz().length() < 0.05 {
                velocity.0.x = 0.0;
                velocity.0.z = 0.0;
            }
        }
    }
}

// GPU-based particle physics system
fn gpu_particle_physics(
    player_query: Query<(&Transform, &Velocity), With<Player>>,
    particle_query: Query<&Transform, (With<RegolithParticle>, Without<Player>)>,
    time: Res<Time>,
) {
    // This system interfaces with the GPU compute shader
    // The actual GPU work is handled by the GpuComputePlugin systems
    
    if let Ok((player_transform, player_velocity)) = player_query.single() {
        let particle_count = particle_query.iter().count();
        
        // Prepare uniforms for GPU compute
        let _uniforms = ComputeUniforms {
            delta_time: time.delta_secs(),
            gravity: LUNAR_GRAVITY,
            particle_count: particle_count as u32,
            ground_level: MIN_PARTICLE_RADIUS,
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
        
        // The GPU compute is dispatched automatically by the GpuComputePlugin
        // This system mainly serves as a coordination point
    }
}

// System to monitor particle stability and detect excessive jittering
fn monitor_particle_stability(
    particle_query: Query<&Velocity, With<RegolithParticle>>,
    time: Res<Time>,
) {
    // Only run this check every 2 seconds to avoid performance impact
    if time.elapsed_secs() % 2.0 < time.delta_secs() {
        let mut high_frequency_particles = 0;
        let mut micro_motion_particles = 0;
        let mut stationary_particles = 0;
        let mut total_kinetic_energy = 0.0;
        let particle_count = particle_query.iter().count();
        
        for velocity in particle_query.iter() {
            let speed = velocity.0.length();
            total_kinetic_energy += speed * speed;
            
            // Count particles with different motion types
            if speed < MIN_VELOCITY_THRESHOLD {
                stationary_particles += 1;
            } else if speed > 0.05 && speed < 0.3 {
                micro_motion_particles += 1;
            } else if speed > 0.3 {
                high_frequency_particles += 1;
            }
        }
        
        let avg_kinetic_energy = if particle_count > 0 {
            total_kinetic_energy / particle_count as f32
        } else {
            0.0
        };
        
        let jitter_percentage = (micro_motion_particles as f32 / particle_count as f32) * 100.0;
        let stationary_percentage = (stationary_particles as f32 / particle_count as f32) * 100.0;
        
        // Log stability metrics
        println!("Particle stability: {:.1}% stationary, {:.1}% micro-motion, {} active, avg KE: {:.4}",
                 stationary_percentage, jitter_percentage, high_frequency_particles, avg_kinetic_energy);
        
        // Only warn if jittering is excessive
        if jitter_percentage > 50.0 {
            println!("WARNING: High jittering detected - {:.1}% of particles have micro-motion", jitter_percentage);
        } else if jitter_percentage < 10.0 {
            println!("SUCCESS: Low jittering - only {:.1}% of particles have micro-motion", jitter_percentage);
        }
    }
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
    // Create UI camera
    commands.spawn(Camera2d);
    
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