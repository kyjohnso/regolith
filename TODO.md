# Regolith - Lunar Simulation Engine TODO

## Project Overview
Building a game/simulation engine using Rust and Bevy 0.16.1 that allows players to move around a realistic lunar surface and interact with regolith spheroids. The goal is to explore the remnants of a moon-wide compute farm and use regolith to rebuild, adapt, and evolve it.

**Technical Goals:**
- Explore limits of Bevy ECS
- Implement GPU-based physics for regolith simulation
- Target 1M+ particles with real-time physics at 30fps minimum
- Create sandbox for lunar computation exploration

## Development Progress

### Phase 1: Foundation & Basic Setup
- [x] Set up Rust project with Bevy 0.16.1 and basic dependencies
- [x] Create basic Bevy app with 3D scene and orbit camera controls
- [x] Test: Run cargo run to verify basic app launches and camera works
- [x] Add lunar surface terrain (simple plane or heightmap)
- [x] **UPDATED**: Implemented hilly terrain with procedural generation using sine waves
- [x] Test: Run cargo run to verify terrain renders correctly

### Phase 2: Player Movement & Interaction
- [x] Implement basic player movement with lunar gravity (1/6th Earth)
- [x] Test: Run cargo run to verify player movement feels right

### Phase 3: Basic Particle System
- [x] Add simple regolith particle spawning system (start with ~1000 particles)
- [x] Create basic particle rendering system with instanced meshes
- [x] Test: Run cargo run to verify particles spawn and render
- [x] Implement CPU-based sphere-sphere collision detection for validation
- [x] Test: Run cargo run to verify collision detection works

### Phase 4: CPU Performance Optimization
- [ ] Optimize CPU-based particle physics performance
- [ ] Scale up to 10k particles with CPU optimization and profiling
- [ ] Test: Run cargo run and profile performance at 10k particles
- [ ] Implement advanced CPU-based spatial hashing for collision detection
- [ ] Optimize CPU memory management and data structures
- [ ] Scale up to 100k particles with CPU performance tuning
- [ ] Test: Run cargo run and measure fps at 100k particles

### Phase 5: Advanced Physics & Interactions (CPU-based)
- [x] Implement particle interaction with terrain/surface (Updated for hilly terrain)
- [x] Test: Run cargo run to verify terrain-particle interactions
- [ ] Add basic particle clustering and cohesion effects (CPU implementation)
- [ ] Implement advanced particle interaction mechanics
- [ ] Test: Run cargo run to verify advanced physics interactions

### Phase 6: Game Mechanics
- [x] Add particle size variation and basic shape differentiation
- [x] Implement player-regolith interaction mechanics
- [x] Test: Run cargo run to verify player can interact with particles
- [ ] Create foundation for compute farm remnants system
- [ ] Add basic regolith manipulation tools for rebuilding

### Phase 7: GPU Compute Architecture (Future Implementation)
- [x] Design GPU compute shader architecture for particle physics
- [x] Implement wgpu compute shader for particle position updates (Infrastructure only)
- [ ] Complete GPU-CPU data synchronization for full GPU physics pipeline
- [ ] Test: Run cargo run to verify GPU compute integration (GPU physics working)
- [ ] Add GPU-based spatial hashing for efficient collision detection
- [ ] Implement GPU sphere-sphere collision response system
- [ ] Test: Run cargo run to verify GPU physics pipeline
- [ ] Scale up to 1M+ particles with GPU implementation at target 30fps performance
- [ ] Test: Run cargo run and validate 1M+ particle performance with GPU

## Technical Architecture

### Core Systems
- **ECS Architecture**: Leveraging Bevy's Entity Component System
- **Rendering**: Instanced mesh rendering for particles
- **Physics**: CPU-based physics (optimized), with future GPU compute shader migration
- **Collision Detection**: Optimized CPU spatial hashing, future GPU implementation
- **Camera**: Pan-orbit camera for exploration

### Performance Targets
- **Current Focus**: Optimize CPU performance for 100k+ particles
- **Future Goal**: 1M+ particles with GPU compute
- **Frame Rate**: 30fps minimum
- **Platform**: Desktop (Windows/Linux/macOS)
- **Migration Path**: CPU optimization → GPU compute shaders

### Key Dependencies
- `bevy = "0.16.1"`
- `bevy_panorbit_camera = "0.28.0"`
- `rand = "0.8"`

## Notes
- Start simple with sphere-sphere collisions and basic gravity
- Add complexity iteratively
- Regular testing with `cargo run` after each major addition
- **Current Focus**: Optimize CPU-based physics before GPU migration
- GPU implementation available on separate branch for future integration

## Recent Updates
### Hilly Terrain Implementation (Completed)
- **Replaced flat plane with procedural hilly terrain** using multiple sine wave functions
- **Added terrain height calculation function** for consistent collision detection
- **Updated collision systems** for both player and particles to work with variable terrain heights
- **Maintains performance** with 5000 particles settling naturally on hills and valleys
- **Enhanced visual realism** with proper normal calculation for lighting on terrain