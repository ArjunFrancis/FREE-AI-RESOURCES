# World Models

**World Models** are AI systems that learn internal representations of how the physical world works—understanding physics, spatial relationships, object dynamics, and cause-and-effect. Unlike traditional AI that learns pattern recognition, world models can **simulate** future states, enabling them to plan, reason, and interact intelligently with complex environments. They represent a critical step toward **Artificial General Intelligence (AGI)**.

## 🎯 Overview

**What Are World Models?**

Imagine an AI that doesn't just recognize objects in images, but understands:
- How objects will move if pushed
- What happens when things collide
- How gravity affects different materials
- What the other side of a building looks like
- How scenes change over time

World models build these **mental simulations** of reality. They're the key to:
- **Robots** that navigate unfamiliar spaces
- **Autonomous vehicles** that predict traffic
- **AI agents** that solve complex real-world problems  
- **Game engines** that generate infinite interactive worlds
- **Scientific simulations** for weather, climate, and physics

**Why They Matter**: Current AI (like ChatGPT) excels at language but lacks physical understanding. World models bridge this gap, enabling AI to interact with the real world as intelligently as humans do.

---

## 📋 Topics Covered

- **Generative World Models**: Creating interactive 3D environments
- **Physics Simulation**: Understanding gravity, collision, dynamics, friction
- **Spatial Reasoning**: 3D scene understanding and prediction
- **Temporal Dynamics**: Predicting how scenes evolve over time
- **Embodied AI**: Training agents in simulated worlds
- **Model-Based Reinforcement Learning**: Planning via simulation
- **Video Generation**: Creating realistic, physically-consistent video
- **Digital Twins**: Virtual replicas of real systems
- **Causal Reasoning**: Understanding cause-effect relationships
- **Path to AGI**: Foundational capability for general intelligence

---

## 🚀 Leading Research & Platforms

### Industry Leaders

#### 1. **Google DeepMind - Genie Series** 🔴 Advanced
- **URL**: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- **Research**: Genie 2 (Dec 2024), Genie 3 (Aug 2025)
- **Description**: DeepMind's **Genie 3** is a breakthrough world model that generates diverse, interactive 3D environments from just a text prompt. Can simulate realistic physics, complex interactions, and maintain stability for extended periods.
- **Key Capabilities**:
  - Generate 3D worlds from text descriptions
  - Interactive environments (jumping, swimming, object manipulation)
  - Emergent physics understanding
  - Multi-minute stable simulations
  - Rich environmental diversity
  - Real-time responsiveness to user actions
- **Training**: Trained on extensive video datasets showing physics, interactions, and environmental dynamics
- **Significance**: Major milestone toward AGI—demonstrates AI can build coherent mental models of reality
- **Applications**: Robot training, game development, autonomous systems, scientific simulation
- **Best For**: Understanding state-of-the-art world models, AGI research, generative simulations

#### 2. **Meta AI - Habitat 3 Platform** 🔴 Advanced
- **URL**: https://aihabitat.org/  
- **GitHub**: https://github.com/facebookresearch/habitat-sim
- **Description**: Meta's open-source platform for training embodied AI agents in realistic 3D environments. Habitat 3 enables robots to learn navigation, manipulation, and human interaction safely through simulation before real-world deployment.
- **Key Features**:
  - High-speed 3D simulation (10,000+ FPS)
  - Photorealistic rendering
  - Physics engine integration
  - Human-robot interaction simulation
  - Multi-agent environments
  - Transfer learning to real robots
  - Supports VR headsets
- **Open Source**: Fully free with extensive documentation
- **Research Papers**: 100+ publications using Habitat
- **Applications**: Robot navigation, object manipulation, human-AI collaboration
- **Best For**: Training embodied AI, robotics research, sim-to-real transfer

#### 3. **NVIDIA Cosmos - World Foundation Models** 🔴 Advanced
- **URL**: https://www.nvidia.com/en-us/glossary/world-models/
- **URL**: https://www.ibm.com/think/news/cosmos-ai-world-models (IBM Partnership)
- **Description**: NVIDIA's **Cosmos** is an open-source world foundation model platform for generating physically-accurate simulations at scale. Powers robot training, autonomous vehicles, and industrial automation with synthetic data that obeys real-world physics.
- **Key Technologies**:
  - Physics-accurate world generation
  - Isaac Sim integration (robot simulation)
  - Synthetic training data generation
  - Power-efficient inference (Jetson edge devices)
  - Multi-resolution downscaling
  - Customizable with proprietary data
- **Open Source**: Available under open model license
- **Enterprise Adoption**: Uber, Figure AI, Waabi (autonomous vehicles)
- **Applications**: 
  - Factory robots (warehouse simulations)
  - Self-driving cars (traffic scenarios)
  - Industrial automation
  - Weather forecasting (IBM partnership)
- **Best For**: Enterprise applications, robot simulation, autonomous systems, synthetic data

#### 4. **World Labs - Fei-Fei Li's Vision** 🔴 Advanced
- **URL**: https://drfeifei.substack.com/p/from-words-to-worlds-spatial-intelligence
- **Founded**: 2024 by Fei-Fei Li (Stanford, ImageNet creator)
- **Description**: World Labs is building next-generation world models that understand semantically, physically, geometrically, and dynamically complex 3D environments. Mission: Enable AI to perceive and interact with the world as richly as humans do.
- **Research Focus**:
  - Foundational world model architecture
  - Large-scale 3D training data
  - Beyond 1D/2D sequence modeling
  - Multimodal spatial understanding
  - Embodied AI and robotics
  - Scientific simulation
- **Key Insight**: "World models are the defining challenge of the next decade in AI—bridging the gap between language AI and physical AI."
- **Funding**: $230M Series A (June 2024)
- **Team**: Leading researchers from Stanford, Google, Meta
- **Why It Matters**: Led by the visionary behind ImageNet (which catalyzed deep learning revolution)
- **Best For**: Understanding future direction of AI, foundational research, AGI path

#### 5. **Yann LeCun - Meta JEPA Architecture** 🔴 Advanced
- **URL**: https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/ (I-JEPA)
- **Description**: Meta's Chief AI Scientist Yann LeCun describes world models as systems that observe environments and predict outcomes while accounting for unknowns. His **Joint Embedding Predictive Architecture (JEPA)** learns abstract representations for efficient world modeling.
- **Key Concepts**:
  - Prediction in abstract representation space (not pixel-level)
  - Energy-based models
  - Self-supervised learning from video
  - Handling uncertainty and unknowns
  - Efficient computation
- **I-JEPA**: Image-based JEPA released in 2023, learns world representations from images
- **V-JEPA**: Video extension (2024)
- **Research**: Multiple papers on world models and predictive learning
- **Why It Matters**: Leading theoretical foundation for world models from Turing Award winner
- **Best For**: Understanding world model theory, self-supervised learning, energy-based models

### Open-Source & Research Tools

#### 6. **OpenAI - World Models Research** 🟡 Intermediate | 🔴 Advanced
- **URL**: https://openai.com/research/ (Search: world models, simulation)
- **Description**: OpenAI's research into world models for reinforcement learning and agent training. Notable work includes learning compact world representations for model-based RL.
- **Key Research**:
  - World Models paper (Ha & Schmidhuber, 2018) - foundational work
  - Model-based RL with world models
  - Generative modeling of environments
  - Video prediction models
- **Applications**: Game-playing agents, robotic control, simulated environments
- **Publications**: Available on arXiv and OpenAI blog
- **Best For**: Understanding world model foundations, RL applications, academic research

#### 7. **IBM Research - Prithvi World Models** 🟡 Intermediate | 🔴 Advanced  
- **URL**: https://www.ibm.com/think/news/cosmos-ai-world-models
- **URL**: https://research.ibm.com/blog/foundation-models-climate (Prithvi Climate)
- **Description**: IBM's **Prithvi** foundation models learn physical dynamics of global atmospheric systems. Demonstrates world models for climate and weather forecasting with multi-granular predictions and downscaling.
- **Key Features**:
  - Physics-compliant simulations
  - Global atmospheric modeling
  - Multi-resolution forecasting
  - Satellite data integration
  - Climate scenario modeling
- **Applications**: Weather prediction, climate modeling, disaster response, agriculture
- **Open Source**: Prithvi models available on Hugging Face
- **Significance**: Shows world models' impact beyond robotics—applicable to Earth systems
- **Best For**: Climate AI, weather forecasting, scientific applications, geospatial modeling

### Educational Resources & Guides

#### 8. **Built In - "Move Over LLMs: World Models Explained"** 🟢 Beginner | 🟡 Intermediate
- **URL**: https://builtin.com/articles/ai-world-models-explained
- **Description**: Comprehensive beginner-friendly guide to world models. Explains how they differ from LLMs, why they matter for AGI, and real-world applications in accessible language.
- **Topics Covered**:
  - What are world models?
  - How do they work?
  - LLMs vs world models
  - Path to AGI
  - Current limitations
  - Industry applications
- **Key Insight**: "World models enable AI to learn general principles of how the world works, not memorize step-by-step instructions."
- **Best For**: Beginners understanding world models, non-technical overview, AGI concepts

#### 9. **Forbes - "The Next Giant Leap for AI"** 🟢 Beginner | 🟡 Intermediate
- **URL**: https://www.forbes.com/sites/bernardmarr/2025/12/08/the-next-giant-leap-for-ai-is-called-world-models/
- **Author**: Bernard Marr (Technology futurist)
- **Description**: Industry analysis of world models' transformative potential. Covers Google Genie 3, Meta Habitat 3, and how world models will revolutionize robotics, gaming, and simulation.
- **Key Points**:
  - Two approaches: dynamic real-time vs pre-generated environments
  - Industry leaders (Google, Meta, NVIDIA)
  - Applications in virtual worlds, metaverse, training AI
  - Market implications
- **Best For**: Business perspective, industry trends, executive overview

#### 10. **DeepFA - "World Models: Key to Achieving AGI"** 🟡 Intermediate | 🔴 Advanced
- **URL**: https://deepfa.ir/en/blog/world-model-ai-agi-future
- **Description**: In-depth technical guide to world model architectures, implementations, and the path to AGI. Covers VAEs, Transformers, Diffusion Models for world modeling.
- **Topics**:
  - **Architectures**: VAE-based, Transformer-based, Diffusion-based world models
  - **Implementation methods**: Code examples and frameworks
  - **Genie series deep-dive**: Technical analysis of Google's approach
  - **Practical applications**: Robotics, games, VR, simulation, prediction
  - **Future challenges**: Computational complexity, data requirements, multimodal integration
- **Code Examples**: PyTorch implementations
- **Best For**: Technical practitioners, architecture comparison, implementation guidance

### Academic Research & Surveys

#### 11. **World Models: A Survey (ACM Computing Surveys 2024)** ⭐ **NEW** 🔴 Advanced
- **URL**: https://dl.acm.org/doi/10.1145/3689031
- **Authors**: Marc Rigter et al. (Oxford, Imperial College)
- **Published**: ACM Computing Surveys (CSUR), September 2024
- **Description**: Comprehensive academic survey of world models across deep learning, reinforcement learning, and cognitive science. Reviews 200+ papers covering model architectures (Transformer-based, diffusion models, graph neural networks), learning methods (self-supervised, model-based RL), and applications in robotics, autonomous driving, and scientific simulation. Analyzes current limitations and future research directions.
- **Key Topics**:
  - World model architectures and taxonomies
  - Learning algorithms (supervised, self-supervised, RL-based)
  - Evaluation benchmarks and metrics
  - Transfer learning and generalization
  - Computational efficiency and scalability
  - Multimodal world models (vision + language + physics)
  - Applications: robotics, AVs, games, science
- **Scope**: 200+ papers reviewed (2015-2024)
- **Pages**: Comprehensive survey paper
- **Significance**: Authoritative academic review from top-tier CS journal
- **Best For**: Researchers, PhD students, comprehensive understanding of field, literature review
- [Tags: `survey` `academic` `acm-csur` `oxford` `world-models` `rl` `deep-learning` `2024`]

#### 12. **ICLR 2025 Workshop on World Models** ⭐ **NEW** 🔴 Advanced
- **URL**: https://iclr.cc/virtual/2025/workshop/worldmodels (Check ICLR 2025 conference site)
- **Description**: Official ICLR 2025 workshop dedicated to world models featuring cutting-edge research papers, invited talks from leading researchers (DeepMind, Meta, OpenAI), and panel discussions on future directions. Topics include generative world models, video prediction, model-based RL, sim-to-real transfer, and scaling laws for world models.
- **Key Features**:
  - Peer-reviewed workshop papers
  - Invited talks from industry leaders
  - Poster sessions with latest research
  - Panel discussions on AGI path
  - Open problems and challenges
- **Topics Covered**:
  - Generative world models (Genie-style)
  - Scalable video prediction
  - Physics-informed world models
  - Multimodal understanding
  - Efficient training methods
  - Evaluation benchmarks
- **Access**: Workshop papers typically free on OpenReview
- **Significance**: Top-tier ML conference workshop showcasing newest research
- **Best For**: Cutting-edge research, state-of-the-art methods, networking with researchers
- [Tags: `workshop` `iclr` `conference` `research-papers` `world-models` `2025`]

### Community Resources

#### 13. **Awesome World Models (GitHub Curated List)** ⭐ **NEW** 🟡 Intermediate | 🔴 Advanced
- **URL**: https://github.com/topics/world-models (GitHub Topic)
- **Alternative**: Search GitHub for "awesome-world-models" repositories
- **Description**: Community-curated collection of world model papers, code implementations, datasets, and tutorials. Includes links to foundational papers (Ha & Schmidhuber 2018), recent breakthroughs (Genie, DreamerV3), open-source implementations in PyTorch/TensorFlow, benchmark environments, and educational resources. Actively maintained by the research community.
- **Contents**:
  - 📄 **Papers**: Foundational + recent research organized by topic
  - 💻 **Code**: Open-source implementations (PyTorch, JAX, TensorFlow)
  - 🗂️ **Datasets**: Training data for world models
  - 🎯 **Benchmarks**: Evaluation environments (DeepMind Control, Atari)
  - 📚 **Tutorials**: Blog posts, videos, courses
  - 🏢 **Industry Projects**: Genie, Habitat, JEPA, DreamerV3
- **Popular Implementations**:
  - World Models (original Ha & Schmidhuber)
  - DreamerV3 (model-based RL)
  - TransDreamer (Transformer world models)
  - IRIS (Imagination with auto-regressive models)
- **Best For**: Discovering implementations, finding papers, hands-on learning, starting research
- [Tags: `github` `awesome-list` `code` `papers` `implementations` `community` `world-models`]

---

## 📚 Key Concepts

### Mental Simulation
World models create internal "mental" representations of environments, enabling AI to simulate "what would happen if..." scenarios without physical trial-and-error.

### Model-Based vs Model-Free RL
- **Model-Free**: Learn actions directly from rewards (e.g., AlphaGo)
- **Model-Based**: Learn a world model first, then plan actions by simulating outcomes (more sample-efficient)

### Generative World Models  
Can generate novel, realistic environments (like Genie 3). Go beyond prediction to creation of coherent 3D worlds.

### Physics Understanding
Learn implicit physics: gravity, collision, object permanence, material properties, cause-effect relationships.

### Temporal Coherence
Maintain consistency over time—objects don't teleport, scenes evolve smoothly, physics laws remain constant.

### Sim-to-Real Transfer
Train AI in simulated world models, then deploy to real world. Safer, faster, cheaper than real-world training.

---

## 🎯 Applications

**Robotics**: Train robots in simulation before real-world deployment  
**Autonomous Vehicles**: Predict traffic, pedestrian behavior, edge cases  
**Game Development**: Generate infinite interactive game worlds  
**Scientific Simulation**: Climate, weather, physics, materials science  
**AR/VR**: Create immersive, physically-consistent virtual worlds  
**Digital Twins**: Virtual replicas of factories, cities, systems  
**Urban Planning**: Simulate city growth, traffic, infrastructure  
**Drug Discovery**: Simulate molecular interactions  
**Education**: Interactive physics simulations for learning  
**Content Creation**: Automated 3D environment generation  

---

## 🔗 Related Categories

- [Spatial Intelligence](spatial-intelligence.md) - 3D understanding and reasoning
- [Reinforcement Learning](reinforcement-learning.md) - Model-based RL
- [Robotics & Embodied AI](robotics-embodied-ai.md) - Physical AI training
- [Generative AI](generative-ai.md) - Content generation
- [Computer Vision](computer-vision.md) - Visual perception
- [Physics Simulation](physics-simulation.md) - Dynamics modeling

### University Resources  
- [Stanford AI Resources](stanford-ai-resources.md) - Fei-Fei Li's research
- [MIT AI Resources](mit-ai-resources.md) - Embodied AI
- [Berkeley AI Resources](berkeley-ai-resources.md) - Model-based RL

---

## 📊 Statistics

**Resource Count**: 13 platforms, research groups, and guides (+3 new)  
**Market Impact**: Critical path to AGI  
**Key Players**: Google DeepMind, Meta, NVIDIA, World Labs, OpenAI, IBM  
**Research Hubs**: Stanford, MIT, Berkeley, CMU, ETH Zurich, Oxford, Imperial College  
**Investment**: $500M+ in world model startups (2024-2025)  
**Last Updated**: February 28, 2026  

---

## 💡 Learning Path

**Beginners**:
1. Read Built In guide for overview
2. Watch Google Genie 3 demos
3. Explore Forbes article for applications
4. Browse Awesome World Models GitHub for curated resources

**Intermediate**:
1. Study DeepFA technical guide
2. Read Yann LeCun's JEPA papers
3. Experiment with Meta Habitat (free)
4. Explore code implementations on Awesome World Models
5. Review ICLR 2025 workshop papers

**Advanced**:
1. Read ACM CSUR comprehensive survey (200+ papers)
2. Implement VAE-based world model
3. Study Fei-Fei Li's World Labs vision
4. Explore NVIDIA Cosmos
5. Read foundational papers (Ha & Schmidhuber, 2018)
6. Build custom world model for specific domain
7. Attend ICLR workshop sessions

---

## 🗓️ Timeline

**2018**: World Models paper (Ha & Schmidhuber) - foundational work  
**2020-2022**: Model-based RL gains traction  
**2023**: I-JEPA (Meta), early generative world models  
**2024**: Genie 2 (Google), Habitat 3 (Meta), Prithvi (IBM)  
**2024**: World Labs founded ($230M funding)  
**2024**: ACM CSUR comprehensive survey published  
**2025**: **Genie 3** breakthrough - text-to-world generation  
**2025**: NVIDIA Cosmos open-source release  
**2025**: ICLR Workshop on World Models  
**2026+**: Expected mainstream adoption in robotics, AVs, gaming  

---

### Contributing

To add a resource:

✅ **World Model Focus**: Must relate to simulating environments, physics, or dynamics  
✅ **Free Access**: Research papers, open-source code, or free documentation  
✅ **Technical Depth**: Should explain architecture, methodology, or applications  
✅ **Reputable Source**: Academic institutions, leading AI labs, industry leaders  

**Format**:
```markdown
- [Resource Name](URL) - Description emphasizing world modeling capabilities and unique approach.
```

**Sources**: Google DeepMind, Meta AI, NVIDIA, World Labs, IBM, OpenAI, ACM, ICLR, Academic institutions (2024-2026)

---

[← Back to Main README](../README.md)
