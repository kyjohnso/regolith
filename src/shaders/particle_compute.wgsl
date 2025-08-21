// Particle compute shader for regolith simulation
// This shader handles particle physics including gravity, collisions, and player interactions

struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    radius: f32,
    mass: f32,
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

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: ComputeUniforms;

// Constants
const LUNAR_GRAVITY: f32 = -1.62;
const RESTITUTION: f32 = 0.2;
const FRICTION: f32 = 0.95;
const MIN_VELOCITY: f32 = 0.001;

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
    
    // Ground collision
    if (particle.position.y <= particle.radius) {
        particle.position.y = particle.radius;
        particle.velocity.y = abs(particle.velocity.y) * -RESTITUTION;
        particle.velocity.x *= FRICTION;
        particle.velocity.z *= FRICTION;
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
    
    // Particle-to-particle collisions (simplified for now)
    // Note: This is O(n²) and will be replaced with spatial hashing
    for (var i: u32 = 0u; i < uniforms.particle_count; i++) {
        if (i == index) {
            continue;
        }
        
        let other = particles[i];
        let distance_vec = particle.position - other.position;
        let distance = length(distance_vec);
        let min_dist = particle.radius + other.radius;
        
        if (distance < min_dist && distance > 0.0) {
            let normal = normalize(distance_vec);
            let overlap = min_dist - distance;
            
            // Separate particles
            particle.position += normal * (overlap * 0.5);
            
            // Simple collision response
            let relative_velocity = particle.velocity - other.velocity;
            let velocity_along_normal = dot(relative_velocity, normal);
            
            if (velocity_along_normal > 0.0) {
                continue;
            }
            
            let impulse_scalar = -(1.0 + RESTITUTION) * velocity_along_normal;
            let impulse = normal * impulse_scalar * 0.5;
            
            particle.velocity -= impulse;
        }
    }
    
    // Apply damping to simulate energy loss
    particle.velocity *= 0.98;
    
    // Clamp very small velocities to zero
    if (length(particle.velocity) < MIN_VELOCITY) {
        particle.velocity = vec3<f32>(0.0, 0.0, 0.0);
    }
    
    // Write back the updated particle
    particles[index] = particle;
}