# Regolith 🌙

**A Lunar Physics Simulation Prototype**

> ⚠️ **This is an early prototype** of what will eventually become a much larger and more ambitious lunar exploration game. The current version demonstrates core physics mechanics and serves as a foundation for future development.

## Overview

Regolith is a 3D physics-based simulation that recreates the unique environment of the lunar surface. Players control a spherical character navigating through procedurally generated lunar terrain covered in thousands of individual regolith (moon dust) particles, all governed by realistic physics including lunar gravity (1/6th of Earth's gravity).

![Gameplay Demo](images/gameplay_demo_compressed.gif)

*Early gameplay demonstration showing physics-based movement and particle interactions*

## Current Features

### 🌍 **Realistic Lunar Physics**
- **Lunar Gravity**: Authentic 1/6th Earth gravity simulation (-2.6 m/s²)
- **Mass-Based Physics**: Player sphere with realistic 70kg mass
- **Angular Momentum Movement**: Roll-based locomotion using WASD keys
- **Dynamic Particle System**: 10,000+ individual regolith particles with varied sizes and masses

### 🎮 **Dual Camera System**
- **Pan-Orbit Camera**: Free-roaming camera for overview perspective
- **First-Person Camera**: Player-following camera with orbital controls
- **Seamless Switching**: Toggle between modes with 'C' key

### 🏔️ **Procedural Terrain Generation**
- **Randomized Landscapes**: Multi-layer noise generation with customizable seeds
- **Varied Topography**: Hills, valleys, and surface details
- **Collision-Accurate**: Full physics collision with generated terrain
- **Reproducible**: Use `--seed <number>` for consistent terrain generation

### ⚙️ **Advanced Physics Engine**
- **Rapier3D Integration**: High-performance 3D physics simulation
- **Particle Interactions**: Individual collision detection for thousands of particles
- **Material Properties**: Realistic friction, restitution, and damping values
- **Dynamic Scaling**: Real-time player size adjustment with physics updates

![Screenshot](images/screenshot_wide.png)

*Wide view of the procedurally generated lunar landscape with particle system*

## Controls

### Movement
- **WASD**: Roll the player sphere (camera-relative movement)
- **Space**: Jump with realistic lunar gravity
- **[ / ]**: Decrease/Increase player scale

### Camera
- **C**: Toggle between Pan-Orbit and First-Person camera modes
- **Arrow Keys** (First-Person): Orbit around player and adjust elevation
- **Mouse Scroll** (First-Person): Zoom in/out
- **Mouse** (Pan-Orbit): Standard orbit camera controls

![Jump Screenshot](images/screenshot_jump.png)

*Player sphere mid-jump, demonstrating lunar gravity physics*

## Technical Implementation

### Built With
- **[Bevy Engine](https://bevyengine.org/)** (0.16.1) - Modern Rust game engine
- **[Rapier3D](https://rapier.rs/)** - High-performance physics simulation
- **Rust** - Systems programming language for performance and safety

### Key Systems
- **Procedural Terrain**: Multi-layer sine wave generation with randomization
- **Particle Physics**: Mass-scaled spherical particles with realistic materials
- **Camera Management**: Dual-camera system with smooth transitions
- **Performance Monitoring**: Real-time FPS tracking and display
- **Debug Systems**: Player position logging and respawn mechanics

## Future Vision

This prototype represents the foundation for a much larger lunar exploration game that will eventually include:

- **🚀 Spacecraft and Vehicles**: Lunar rovers, landers, and orbital mechanics
- **🏗️ Base Building**: Construct and manage lunar outposts
- **🔬 Scientific Missions**: Realistic lunar geology and sample collection
- **🌌 Expanded Environments**: Multiple lunar locations and celestial bodies
- **👥 Multiplayer**: Collaborative lunar exploration and construction
- **📊 Resource Management**: Mining, processing, and logistics systems
- **🎯 Mission Objectives**: Story-driven exploration and discovery

## Getting Started

### Prerequisites
- Rust (latest stable version)
- Git

### Installation
```bash
git clone https://github.com/yourusername/regolith.git
cd regolith
cargo run --release
```

### Command Line Options
```bash
# Run with default terrain seed (42)
cargo run --release

# Run with custom terrain seed
cargo run --release -- --seed 12345
```

## Performance

The simulation handles 10,000+ physics objects simultaneously while maintaining smooth performance. The engine is optimized for:
- **Efficient Collision Detection**: Spatial partitioning for particle interactions
- **Adaptive Physics**: Sleeping particles to reduce computational load
- **Optimized Rendering**: Level-of-detail and frustum culling
- **Memory Management**: Rust's zero-cost abstractions and memory safety

## Development Status

**Current Version**: 0.1.0 (Early Prototype)

This is a proof-of-concept demonstrating core physics mechanics and rendering systems. The codebase is actively being developed and expanded toward the full vision outlined above.

### Recent Additions
- ✅ Dual camera system with smooth transitions
- ✅ Real-time player scaling with physics updates
- ✅ Procedural terrain generation with seeded randomization
- ✅ Performance monitoring and FPS display
- ✅ Particle material differentiation based on size

### Next Milestones
- 🔄 Enhanced particle interactions and dust cloud effects
- 🔄 Improved terrain generation with more realistic lunar features
- 🔄 Basic vehicle physics for lunar rovers
- 🔄 Simple construction mechanics for habitat modules

## Contributing

As this project evolves from prototype to full game, contributions will be welcomed. The current focus is on establishing core systems and architecture that will support the expanded feature set.

## License

This project is currently in early development. License terms will be established as the project matures.

---

*Regolith - Exploring the Moon, One Particle at a Time* 🌙✨