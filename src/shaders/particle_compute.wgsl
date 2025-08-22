// Particle compute shader for regolith simulation
// This shader handles particle physics including gravity, collisions, and player interactions

struct Particle {
    position: vec3<f32>,
    _padding1: f32,
    velocity: vec3<f32>,
    _padding2: f32,
    radius: f32,
    mass: f32,
    _padding3: vec2<f32>,
}

struct ComputeUniforms {
    delta_time: f32,
    gravity: f32,
    particle_count: u32,
    ground_level: f32,
    player_position: vec3<f32>,
    player_radius: f32,
    player_velocity: vec3<f32>,
}

struct SpatialHashUniforms {
    grid_size: vec3<u32>,
    cell_size: f32,
    world_min: vec3<f32>,
    world_max: vec3<f32>,
}

struct HashCell {
    particle_count: u32,
    particle_indices: array<u32, 32>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: ComputeUniforms;
@group(0) @binding(2) var<uniform> spatial_uniforms: SpatialHashUniforms;
@group(0) @binding(3) var<storage, read> spatial_hash: array<HashCell>;

// Constants
const LUNAR_GRAVITY: f32 = -1.62;
const RESTITUTION: f32 = 0.2;
const FRICTION: f32 = 0.95;
const MIN_VELOCITY: f32 = 0.08;
const VELOCITY_DAMPING: f32 = 0.95;
const COLLISION_DAMPING: f32 = 0.6;
const GROUND_STABILITY_THRESHOLD: f32 = 0.1;
const GROUND_DAMPING: f32 = 0.85;
const PARTICLE_RESTITUTION: f32 = 0.4;
const PARTICLE_FRICTION: f32 = 0.8;
const SEPARATION_FORCE: f32 = 0.9;

// Hash function for 3D position to grid cell
fn hash_position(pos: vec3<f32>) -> vec3<u32> {
    let grid_pos = (pos - spatial_uniforms.world_min) / spatial_uniforms.cell_size;
    return vec3<u32>(
        clamp(u32(grid_pos.x), 0u, spatial_uniforms.grid_size.x - 1u),
        clamp(u32(grid_pos.y), 0u, spatial_uniforms.grid_size.y - 1u),
        clamp(u32(grid_pos.z), 0u, spatial_uniforms.grid_size.z - 1u)
    );
}

// Convert 3D grid coordinates to 1D array index
fn grid_to_index(grid_pos: vec3<u32>) -> u32 {
    return grid_pos.x + grid_pos.y * spatial_uniforms.grid_size.x +
           grid_pos.z * spatial_uniforms.grid_size.x * spatial_uniforms.grid_size.y;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Bounds check
    if (index >= uniforms.particle_count) {
        return;
    }
    
    var particle = particles[index];
    
    // Apply gravity
    particle.velocity.y += LUNAR_GRAVITY * uniforms.delta_time;
    
    // Update position
    particle.position += particle.velocity * uniforms.delta_time;
    
    // Ground collision with improved damping
    if (particle.position.y <= particle.radius) {
        particle.position.y = particle.radius;
        particle.velocity.y = abs(particle.velocity.y) * -RESTITUTION;
        particle.velocity.x *= FRICTION;
        particle.velocity.z *= FRICTION;
        
        // Additional collision damping to prevent bouncing jitter
        particle.velocity *= COLLISION_DAMPING;
        
        // Stop micro-bouncing near ground
        if (abs(particle.velocity.y) < GROUND_STABILITY_THRESHOLD) {
            particle.velocity.y = 0.0;
        }
    }
    
    // Player collision
    let player_distance = distance(particle.position, uniforms.player_position);
    let min_distance = particle.radius + uniforms.player_radius;
    
    if (player_distance < min_distance && player_distance > 0.0) {
        // Calculate collision normal
        let normal = normalize(particle.position - uniforms.player_position);
        
        // Separate particle from player
        let overlap = min_distance - player_distance;
        particle.position += normal * overlap * 0.8;
        
        // Apply player momentum transfer
        let player_speed = length(uniforms.player_velocity);
        if (player_speed > 0.1) {
            let push_force = normalize(uniforms.player_velocity) * (player_speed * 0.5);
            particle.velocity += push_force;
        } else {
            // Minimum push force
            particle.velocity += normal * 2.0;
        }
        
        // Collision response
        let relative_velocity = particle.velocity;
        let velocity_along_normal = dot(relative_velocity, normal);
        
        if (velocity_along_normal < 0.0) {
            let impulse_scalar = -(1.0 + 0.3) * velocity_along_normal;
            let impulse = normal * impulse_scalar;
            particle.velocity += impulse * 0.8;
        }
    }
    
    // Particle-to-particle collisions using spatial hashing
    let particle_grid_pos = hash_position(particle.position);
    
    // Check neighboring cells (3x3x3 = 27 cells)
    for (var dx: i32 = -1; dx <= 1; dx++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dz: i32 = -1; dz <= 1; dz++) {
                let neighbor_grid = vec3<i32>(
                    i32(particle_grid_pos.x) + dx,
                    i32(particle_grid_pos.y) + dy,
                    i32(particle_grid_pos.z) + dz
                );
                
                // Check bounds
                if (neighbor_grid.x < 0 || neighbor_grid.y < 0 || neighbor_grid.z < 0 ||
                    neighbor_grid.x >= i32(spatial_uniforms.grid_size.x) ||
                    neighbor_grid.y >= i32(spatial_uniforms.grid_size.y) ||
                    neighbor_grid.z >= i32(spatial_uniforms.grid_size.z)) {
                    continue;
                }
                
                let neighbor_grid_u = vec3<u32>(
                    u32(neighbor_grid.x),
                    u32(neighbor_grid.y),
                    u32(neighbor_grid.z)
                );
                
                let cell_index = grid_to_index(neighbor_grid_u);
                let cell = spatial_hash[cell_index];
                
                // Check collisions with particles in this cell
                for (var i: u32 = 0u; i < cell.particle_count; i++) {
                    let other_index = cell.particle_indices[i];
                    
                    // Skip self-collision
                    if (other_index == index) {
                        continue;
                    }
                    
                    let other_particle = particles[other_index];
                    let distance_vec = particle.position - other_particle.position;
                    let distance = length(distance_vec);
                    let min_distance = particle.radius + other_particle.radius;
                    
                    // Collision detected
                    if (distance < min_distance && distance > 0.0) {
                        let normal = normalize(distance_vec);
                        let overlap = min_distance - distance;
                        
                        // Separate particles
                        particle.position += normal * overlap * SEPARATION_FORCE * 0.5;
                        
                        // Calculate relative velocity
                        let relative_velocity = particle.velocity - other_particle.velocity;
                        let velocity_along_normal = dot(relative_velocity, normal);
                        
                        // Only resolve if particles are moving towards each other
                        if (velocity_along_normal < 0.0) {
                            // Calculate impulse
                            let total_mass = particle.mass + other_particle.mass;
                            var impulse_magnitude = -(1.0 + PARTICLE_RESTITUTION) * velocity_along_normal;
                            impulse_magnitude *= (2.0 * other_particle.mass) / total_mass;
                            
                            let impulse = normal * impulse_magnitude;
                            particle.velocity += impulse;
                            
                            // Apply friction
                            let tangent_velocity = relative_velocity - normal * velocity_along_normal;
                            particle.velocity -= tangent_velocity * PARTICLE_FRICTION * 0.1;
                        }
                    }
                }
            }
        }
    }
    
    // Apply progressive damping to simulate energy loss
    particle.velocity *= VELOCITY_DAMPING;
    
    // Clamp very small velocities to zero to prevent micro-jittering
    if (length(particle.velocity) < MIN_VELOCITY) {
        particle.velocity = vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // Additional stability for particles near ground
    if (particle.position.y <= particle.radius + 0.02) {
        // Apply extra damping for particles near ground
        particle.velocity *= GROUND_DAMPING;
        
        // Stop vertical jittering near ground
        if (abs(particle.velocity.y) < GROUND_STABILITY_THRESHOLD) {
            particle.velocity.y = 0.0;
        }
        
        // Stop horizontal micro-movements near ground
        if (length(particle.velocity.xz) < 0.05) {
            particle.velocity.x = 0.0;
            particle.velocity.z = 0.0;
        }
    }
    
    // Write back the updated particle
    particles[index] = particle;
}