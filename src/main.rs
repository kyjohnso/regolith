use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use std::collections::{VecDeque, HashMap, HashSet};

mod gpu_compute;
use gpu_compute::{GpuComputePlugin, ComputeUniforms};

// Lunar gravity constant (1/6th of Earth's gravity)
const LUNAR_GRAVITY: f32 = -2.6; //-1.62; // m/s²
const PLAYER_SPEED: f32 =2.0;
const JUMP_IMPULSE: f32 = 2.0;

// Particle system constants
const PARTICLE_COUNT: usize = 10000; // Scale up to test spatial hashing performance
const MIN_PARTICLE_RADIUS: f32 = 0.05; // Smallest particles (fine dust)
const MAX_PARTICLE_RADIUS: f32 = 0.15; // Largest particles (small rocks)
const SPAWN_AREA_SIZE: f32 = 4.0; // Spawn even closer to player for testing

// Damping constants - Balanced to reduce jitter while allowing natural motion
const VELOCITY_DAMPING: f32 = 0.92; // General velocity damping (balanced)
const COLLISION_DAMPING: f32 = 0.5; // Additional damping after collisions (balanced)
const MIN_VELOCITY_THRESHOLD: f32 = 0.1; // Velocities below this are set to zero (balanced threshold)
const GROUND_DAMPING: f32 = 0.8; // Additional damping for particles near ground
const ANGULAR_DAMPING: f32 = 0.9; // For rotational motion if we add it later

// Spatial hashing constants for collision optimization
const SPATIAL_HASH_CELL_SIZE: f32 = 0.5; // Cell size should be roughly 2x max particle radius
const SPATIAL_HASH_TABLE_SIZE: usize = 4096; // Power of 2 for efficient hashing

// Particle sleeping constants for performance optimization - More effective
const SLEEP_VELOCITY_THRESHOLD: f32 = 0.03; // Particles below this velocity can sleep (lower threshold)
const SLEEP_TIME_THRESHOLD: f32 = 0.3; // Time in seconds before particle sleeps (faster sleeping)
const WAKE_DISTANCE_THRESHOLD: f32 = 0.15; // Distance to wake up sleeping particles (smaller wake radius)

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
    is_sleeping: bool,
    sleep_timer: f32,
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
            player_input,
            apply_gravity,
            player_movement,
            particle_physics,
            particle_collisions,
            wake_sleeping_particles,
            player_particle_interactions,
            monitor_particle_stability,
        ));
    }
    
    app.run();
}

// Function to create a hilly terrain mesh
fn create_hilly_terrain(size: f32, resolution: usize) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    
    let step = size / resolution as f32;
    let half_size = size * 0.5;
    
    // Generate height map using simple noise
    let mut heights = vec![vec![0.0; resolution + 1]; resolution + 1];
    for x in 0..=resolution {
        for z in 0..=resolution {
            let world_x = (x as f32 * step) - half_size;
            let world_z = (z as f32 * step) - half_size;
            
            // Create hills using multiple sine waves for natural-looking terrain
            let height =
                (world_x * 0.1).sin() * (world_z * 0.1).cos() * 2.0 +
                (world_x * 0.05).cos() * (world_z * 0.15).sin() * 1.5 +
                (world_x * 0.2).sin() * (world_z * 0.08).cos() * 0.8 +
                (world_x * 0.03).cos() * (world_z * 0.04).sin() * 3.0;
            
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

// Function to get terrain height at a given world position
fn get_terrain_height(x: f32, z: f32) -> f32 {
    // This should match the height calculation in create_hilly_terrain
    (x * 0.1).sin() * (z * 0.1).cos() * 2.0 +
    (x * 0.05).cos() * (z * 0.15).sin() * 1.5 +
    (x * 0.2).sin() * (z * 0.08).cos() * 0.8 +
    (x * 0.03).cos() * (z * 0.04).sin() * 3.0
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

    // Add lunar surface terrain - hilly terrain with lunar-like coloring
    commands.spawn((
        Mesh3d(meshes.add(create_hilly_terrain(100.0, 100))),
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

        // Get terrain height at player position
        let terrain_height = get_terrain_height(transform.translation.x, transform.translation.z);
        let player_ground_level = terrain_height + 0.9; // 0.9 is half the capsule height

        // Terrain-aware ground collision
        if transform.translation.y <= player_ground_level {
            transform.translation.y = player_ground_level;
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
                is_sleeping: false,
                sleep_timer: 0.0,
            },
            Velocity(Vec3::ZERO),
        ));
    }

    println!("Spawned {} regolith particles with varied sizes ({:.3} to {:.3})",
             PARTICLE_COUNT, MIN_PARTICLE_RADIUS, MAX_PARTICLE_RADIUS);
}

// Spatial hash function for 3D coordinates
fn spatial_hash(pos: Vec3) -> usize {
    let x = (pos.x / SPATIAL_HASH_CELL_SIZE).floor() as i32;
    let y = (pos.y / SPATIAL_HASH_CELL_SIZE).floor() as i32;
    let z = (pos.z / SPATIAL_HASH_CELL_SIZE).floor() as i32;
    
    // Simple hash function combining x, y, z coordinates
    let hash = ((x.wrapping_mul(73856093)) ^ (y.wrapping_mul(19349663)) ^ (z.wrapping_mul(83492791))) as usize;
    hash % SPATIAL_HASH_TABLE_SIZE
}

// Get neighboring cell coordinates for collision detection
fn get_neighbor_cells(pos: Vec3) -> Vec<usize> {
    let mut cells = Vec::with_capacity(27); // 3x3x3 = 27 neighboring cells
    
    let base_x = (pos.x / SPATIAL_HASH_CELL_SIZE).floor() as i32;
    let base_y = (pos.y / SPATIAL_HASH_CELL_SIZE).floor() as i32;
    let base_z = (pos.z / SPATIAL_HASH_CELL_SIZE).floor() as i32;
    
    // Check all 27 neighboring cells (including the current cell)
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let x = base_x + dx;
                let y = base_y + dy;
                let z = base_z + dz;
                
                let hash = ((x.wrapping_mul(73856093)) ^ (y.wrapping_mul(19349663)) ^ (z.wrapping_mul(83492791))) as usize;
                cells.push(hash % SPATIAL_HASH_TABLE_SIZE);
            }
        }
    }
    
    cells
}

fn particle_collisions(
    mut particle_query: Query<(&mut Transform, &mut Velocity, &mut RegolithParticle), Without<Player>>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    
    // Collect all particle data with indices, excluding sleeping particles
    let mut particle_data: Vec<(usize, Vec3, Vec3, f32, f32, bool)> = Vec::new();
    let mut sleeping_particles = 0;
    
    for (i, (transform, velocity, particle)) in particle_query.iter().enumerate() {
        if particle.is_sleeping {
            sleeping_particles += 1;
            continue; // Skip sleeping particles for collision detection
        }
        particle_data.push((i, transform.translation, velocity.0, particle.radius, particle.mass, false));
    }
    
    // Build spatial hash table
    let mut spatial_hash_table: HashMap<usize, Vec<usize>> = HashMap::new();
    
    for (i, (_, pos, _, _, _, _)) in particle_data.iter().enumerate() {
        let hash = spatial_hash(*pos);
        spatial_hash_table.entry(hash).or_insert_with(Vec::new).push(i);
    }
    
    // Calculate collision responses using spatial hashing
    let mut velocity_updates: Vec<Vec3> = vec![Vec3::ZERO; particle_data.len()];
    let mut position_updates: Vec<Vec3> = vec![Vec3::ZERO; particle_data.len()];
    let mut collision_pairs: HashSet<(usize, usize)> = HashSet::new();
    
    // Check collisions only within neighboring cells
    for (i, (_, pos_a, vel_a, radius_a, mass_a, _)) in particle_data.iter().enumerate() {
        let neighbor_cells = get_neighbor_cells(*pos_a);
        
        for &cell_hash in &neighbor_cells {
            if let Some(cell_particles) = spatial_hash_table.get(&cell_hash) {
                for &j in cell_particles {
                    if i >= j { continue; } // Avoid duplicate checks and self-collision
                    
                    // Skip if we've already processed this pair
                    if collision_pairs.contains(&(i, j)) || collision_pairs.contains(&(j, i)) {
                        continue;
                    }
                    
                    let (_, pos_b, vel_b, radius_b, mass_b, _) = particle_data[j];
                    
                    let distance = pos_a.distance(pos_b);
                    let min_distance = radius_a + radius_b;
                    
                    // Check for collision
                    if distance < min_distance && distance > 0.0 {
                        collision_pairs.insert((i, j));
                        
                        // Calculate collision normal
                        let normal = (pos_b - *pos_a).normalize();
                        
                        // More aggressive separation to prevent overlap jitter
                        let overlap = min_distance - distance;
                        let separation = normal * (overlap * 0.6); // Increased separation factor
                        
                        position_updates[i] -= separation;
                        position_updates[j] += separation;
                        
                        let total_mass = mass_a + mass_b;
                        
                        // Mass-based elastic collision response
                        let relative_velocity = vel_b - *vel_a;
                        let velocity_along_normal = relative_velocity.dot(normal);
                        
                        // Don't resolve if velocities are separating
                        if velocity_along_normal > 0.0 {
                            continue;
                        }
                        
                        // Restitution (bounciness) - very low for regolith to reduce jitter
                        let restitution = 0.05;
                        let impulse_scalar = -(1.0 + restitution) * velocity_along_normal / total_mass;
                        let impulse = normal * impulse_scalar;
                        
                        // Apply impulse based on mass ratios (heavier particles move less)
                        velocity_updates[i] -= impulse * mass_b;
                        velocity_updates[j] += impulse * mass_a;
                    }
                }
            }
        }
    }
    
    // Apply updates to actual particles and manage sleeping state
    let mut active_particle_index = 0;
    for (mut transform, mut velocity, mut particle) in particle_query.iter_mut() {
        // Skip sleeping particles
        if particle.is_sleeping {
            continue;
        }
        
        // Only apply updates if we have data for this active particle
        if active_particle_index < position_updates.len() {
            transform.translation += position_updates[active_particle_index];
            velocity.0 += velocity_updates[active_particle_index];
            
            // Apply collision damping if there was a collision
            if velocity_updates[active_particle_index].length() > 0.001 {
                velocity.0 *= COLLISION_DAMPING;
                particle.sleep_timer = 0.0; // Reset sleep timer on collision
            }
        }
        
        active_particle_index += 1;
        
        // Apply general velocity damping to simulate energy loss
        velocity.0 *= VELOCITY_DAMPING;
        
        // More aggressive velocity clamping to reduce jitter
        let speed = velocity.0.length();
        if speed < MIN_VELOCITY_THRESHOLD * 0.7 {
            // Clamp small velocities more aggressively
            velocity.0.x *= 0.2;
            velocity.0.z *= 0.2;
            if velocity.0.y.abs() < MIN_VELOCITY_THRESHOLD * 0.4 {
                velocity.0.y *= 0.2;
            }
        }
        
        // Additional micro-jitter elimination
        if speed < MIN_VELOCITY_THRESHOLD * 0.4 {
            velocity.0.x *= 0.1;
            velocity.0.z *= 0.1;
            if velocity.0.y.abs() < MIN_VELOCITY_THRESHOLD * 0.2 {
                velocity.0.y *= 0.1;
            }
        }
        
        // Update sleep timer and check for sleeping
        let final_speed = velocity.0.length();
        if final_speed < SLEEP_VELOCITY_THRESHOLD {
            particle.sleep_timer += dt;
            if particle.sleep_timer > SLEEP_TIME_THRESHOLD {
                particle.is_sleeping = true;
                velocity.0 = Vec3::ZERO; // Stop all movement when sleeping
            }
        } else {
            particle.sleep_timer = 0.0; // Reset timer if moving fast enough
        }
        
        // More aggressive final velocity clamping to reduce micro-jittering
        let final_speed = velocity.0.length();
        if final_speed < MIN_VELOCITY_THRESHOLD * 0.5 {
            velocity.0 = Vec3::ZERO; // Complete stop for small movements
        } else if final_speed < MIN_VELOCITY_THRESHOLD {
            // Zero out horizontal movement, preserve vertical for gravity
            velocity.0.x = 0.0;
            velocity.0.z = 0.0;
            // Keep y-component for gravity unless it's small
            if velocity.0.y.abs() < MIN_VELOCITY_THRESHOLD * 0.3 {
                velocity.0.y = 0.0;
            }
        }
    }
    
    // Print sleeping particle count occasionally for debugging
    if sleeping_particles > 0 && time.elapsed_secs() % 5.0 < dt {
        println!("Sleeping particles: {} / {}", sleeping_particles, particle_query.iter().count());
    }
}

fn player_particle_interactions(
    mut player_query: Query<(&mut Transform, &mut Velocity), With<Player>>,
    mut particle_query: Query<(&mut Transform, &mut Velocity, &mut RegolithParticle), Without<Player>>,
) {
    // Get player data
    if let Ok((mut player_transform, mut player_velocity)) = player_query.single_mut() {
        let player_radius = 0.8; // Increase player interaction radius to push more particles
        let mut collision_count = 0;
        
        // Check collisions with all particles
        for (mut particle_transform, mut particle_velocity, mut particle) in particle_query.iter_mut() {
            // Wake up sleeping particles when player interacts with them
            if particle.is_sleeping {
                let distance = player_transform.translation.distance(particle_transform.translation);
                if distance < player_radius + particle.radius + 0.2 {
                    particle.is_sleeping = false;
                    particle.sleep_timer = 0.0;
                    // Give a small velocity to prevent immediate re-sleeping
                    use rand::Rng;
                    let mut rng = rand::thread_rng();
                    particle_velocity.0 = Vec3::new(
                        (rng.gen::<f32>() - 0.5) * 0.2,
                        0.1,
                        (rng.gen::<f32>() - 0.5) * 0.2,
                    );
                }
                continue; // Skip collision processing for sleeping particles
            }
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
                // BUT preserve gravity effects by only zeroing horizontal components
                if particle_velocity.0.length() < MIN_VELOCITY_THRESHOLD {
                    // Only zero out horizontal movement, preserve vertical (gravity) component
                    particle_velocity.0.x = 0.0;
                    particle_velocity.0.z = 0.0;
                    // Keep y-component for gravity unless it's also very small
                    if particle_velocity.0.y.abs() < MIN_VELOCITY_THRESHOLD * 0.5 {
                        particle_velocity.0.y = 0.0;
                    }
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
    mut particle_query: Query<(&mut Transform, &mut Velocity, &mut RegolithParticle), Without<Player>>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();

    // Apply gravity and physics to active particles only
    for (mut transform, mut velocity, mut particle) in &mut particle_query {
        // Skip sleeping particles for physics calculations
        if particle.is_sleeping {
            continue;
        }
        
        // Apply lunar gravity
        velocity.0.y += LUNAR_GRAVITY * dt;

        // Apply velocity to position
        transform.translation += velocity.0 * dt;

        // Terrain-aware ground collision using individual particle radius
        let terrain_height = get_terrain_height(transform.translation.x, transform.translation.z);
        let particle_ground_level = terrain_height + particle.radius;
        
        if transform.translation.y <= particle_ground_level {
            transform.translation.y = particle_ground_level;
            velocity.0.y = velocity.0.y.abs() * -0.2; // Bounce with energy loss
            velocity.0.x *= 0.95; // Reduced friction - particles slide more on lunar surface
            velocity.0.z *= 0.95; // Reduced friction - particles slide more on lunar surface
            
            // Additional damping after ground collision to prevent bouncing jitter
            velocity.0 *= COLLISION_DAMPING;
            
            // Reset sleep timer on ground collision
            particle.sleep_timer = 0.0;
        }
        
        // Apply general velocity damping
        velocity.0 *= VELOCITY_DAMPING;
        
        // Balanced velocity clamping for physics system
        let speed = velocity.0.length();
        if speed < MIN_VELOCITY_THRESHOLD * 0.5 {
            // Clamp very small velocities more gently
            velocity.0.x *= 0.3;
            velocity.0.z *= 0.3;
            if velocity.0.y.abs() < MIN_VELOCITY_THRESHOLD * 0.3 {
                velocity.0.y *= 0.3;
            }
        }
        
        // Update sleep timer and check for sleeping
        let final_speed = velocity.0.length();
        if final_speed < SLEEP_VELOCITY_THRESHOLD {
            particle.sleep_timer += dt;
            if particle.sleep_timer > SLEEP_TIME_THRESHOLD {
                particle.is_sleeping = true;
                velocity.0 = Vec3::ZERO; // Stop all movement when sleeping
                continue; // Skip further processing for newly sleeping particles
            }
        } else {
            particle.sleep_timer = 0.0; // Reset timer if moving fast enough
        }
        
        // Balanced velocity clamping to reduce micro-jittering
        let final_speed = velocity.0.length();
        if final_speed < MIN_VELOCITY_THRESHOLD * 0.3 {
            velocity.0 = Vec3::ZERO; // Complete stop for very tiny movements
        } else if final_speed < MIN_VELOCITY_THRESHOLD {
            // Zero out horizontal movement, preserve vertical for gravity
            velocity.0.x = 0.0;
            velocity.0.z = 0.0;
            // Keep y-component for gravity unless it's very small
            if velocity.0.y.abs() < MIN_VELOCITY_THRESHOLD * 0.4 {
                velocity.0.y = 0.0;
            }
        }
        
        // Additional stability check: if particle is very close to terrain and moving slowly
        if transform.translation.y <= particle_ground_level + 0.02 {
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

// System to wake up sleeping particles when disturbed
fn wake_sleeping_particles(
    mut particle_query: Query<(&Transform, &mut RegolithParticle, &mut Velocity)>,
    player_query: Query<&Transform, (With<Player>, Without<RegolithParticle>)>,
) {
    if let Ok(player_transform) = player_query.single() {
        // Collect positions of all particles first to avoid borrow checker issues
        let particle_positions: Vec<(usize, Vec3, bool)> = particle_query
            .iter()
            .enumerate()
            .map(|(i, (transform, particle, _))| (i, transform.translation, particle.is_sleeping))
            .collect();
        
        // Check for wake-up conditions
        for (transform, mut particle, mut velocity) in particle_query.iter_mut() {
            if !particle.is_sleeping {
                continue;
            }
            
            let mut should_wake = false;
            
            // Wake up if player is nearby
            let distance_to_player = transform.translation.distance(player_transform.translation);
            if distance_to_player < WAKE_DISTANCE_THRESHOLD * 3.0 {
                should_wake = true;
            }
            
            // Wake up if any active particle is nearby
            for &(_, other_pos, other_sleeping) in &particle_positions {
                if other_sleeping {
                    continue;
                }
                
                let distance = transform.translation.distance(other_pos);
                if distance < WAKE_DISTANCE_THRESHOLD {
                    should_wake = true;
                    break;
                }
            }
            
            if should_wake {
                particle.is_sleeping = false;
                particle.sleep_timer = 0.0;
                // Give a small random velocity to prevent immediate re-sleeping
                use rand::Rng;
                let mut rng = rand::thread_rng();
                velocity.0 = Vec3::new(
                    (rng.gen::<f32>() - 0.5) * 0.1,
                    0.05,
                    (rng.gen::<f32>() - 0.5) * 0.1,
                );
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
        // Note: GPU compute will need to be updated to handle terrain heights
        let terrain_height = get_terrain_height(player_transform.translation.x, player_transform.translation.z);
        let _uniforms = ComputeUniforms {
            delta_time: time.delta_secs(),
            gravity: LUNAR_GRAVITY,
            particle_count: particle_count as u32,
            ground_level: terrain_height + MIN_PARTICLE_RADIUS,
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
    particle_query: Query<(&Velocity, &RegolithParticle), With<RegolithParticle>>,
    time: Res<Time>,
) {
    // Only run this check every 2 seconds to avoid performance impact
    if time.elapsed_secs() % 2.0 < time.delta_secs() {
        let mut high_frequency_particles = 0;
        let mut micro_motion_particles = 0;
        let mut stationary_particles = 0;
        let mut sleeping_particles = 0;
        let mut total_kinetic_energy = 0.0;
        let particle_count = particle_query.iter().count();
        
        for (velocity, particle) in particle_query.iter() {
            let speed = velocity.0.length();
            total_kinetic_energy += speed * speed;
            
            // Count sleeping particles separately
            if particle.is_sleeping {
                sleeping_particles += 1;
                stationary_particles += 1; // Sleeping particles are stationary
            } else {
                // Count particles with different motion types - realistic thresholds
                if speed < 0.02 { // Truly stationary (but not sleeping)
                    stationary_particles += 1;
                } else if speed >= 0.02 && speed < 0.08 { // Visible micro-motion/jitter
                    micro_motion_particles += 1;
                } else if speed >= 0.08 {
                    high_frequency_particles += 1;
                }
            }
        }
        
        let avg_kinetic_energy = if particle_count > 0 {
            total_kinetic_energy / particle_count as f32
        } else {
            0.0
        };
        
        let jitter_percentage = (micro_motion_particles as f32 / particle_count as f32) * 100.0;
        let stationary_percentage = (stationary_particles as f32 / particle_count as f32) * 100.0;
        let sleeping_percentage = (sleeping_particles as f32 / particle_count as f32) * 100.0;
        
        // Log stability metrics with sleeping info
        println!("Particle stability: {:.1}% stationary ({:.1}% sleeping), {:.1}% micro-motion, {} active, avg KE: {:.4}",
                 stationary_percentage, sleeping_percentage, jitter_percentage, high_frequency_particles, avg_kinetic_energy);
        
        // Only warn if jittering is excessive
        if jitter_percentage > 5.0 {
            println!("WARNING: High jittering detected - {:.1}% of particles have micro-motion", jitter_percentage);
        } else if jitter_percentage < 2.0 {
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