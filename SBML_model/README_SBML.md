# Yeast Galactose Switch Hybrid SBML Models

This directory contains System Biology Markup Language (SBML) Level 3 Version 1 models that encode a highly customized hybrid deterministic-stochastic modeling framework for the galactose switch. 

Due to the absence of natively generalized properties in root SBML matching complex attributes like `numpy` lattice masks or multi-solver demarcations, the generated structural architecture explicitly expands upon standard element tags using localized `<annotation>` pipelines. Any hybrid engine seeking to parse these models should extract the supplementary logic according to the conventions defined below.

---

## Part 1: CME-ODE SBML Schema
The CME-ODE model (`galactose_cme_ode.xml`) acts as the foundational non-spatial architecture, coupling mass-action metabolic kinetics with stochastic gene regulatory networks. 

### 1. Initial Abundance
Initial particle counts (representing molecular abundance at t=0) are natively populated directly within the `initialAmount` parameter of each defined `<species>` node. Because this is a stochastic regime, these values exclusively map to discrete integers.

**Example:**
```xml
<species id="G2" compartment="plasmaMembrane" initialAmount="1157" hasOnlySubstanceUnits="true" constant="false"/>
<species id="G4" compartment="nucleoplasm" initialAmount="1" hasOnlySubstanceUnits="true" constant="false"/>
```

### 2. Kinetic Parameters & Reaction Logic
Reaction rates are uncoupled from structural equations. The base values for binding events, decay rates, and transport velocities are securely mapped globally inside `<listOfParameters>`. The actual computational formula binds to these referenced parameters actively inside the specific reaction's `<kineticLaw>`.

**Example:**
```xml
<listOfParameters>
  <!-- Global parameter definition -->
  <parameter id="kcat_GK" value="3350" constant="true"/>
</listOfParameters>

...
<kineticLaw>
  <math xmlns="http://www.w3.org/1998/Math/MathML">
    <!-- Math referencing the explicit kinetic parameter ID -->
    <apply>
      <times/>
      <ci> kcat_GK </ci>
      <ci> G1GAI </ci>
    </apply>
  </math>
</kineticLaw>
```

### 3. Simulation Engine Segregation (CME vs. ODE)
Reactions mapped to massive mass-action transport gradients (Gal2 scaling) fundamentally crash stochastic stepping sequences. Therefore, the internal `<reaction>` tree uses an explicit `simulation_engine` flag to assign numerical dispatching. Continuous ODE integrators strictly filter for `type="ODE"`, while Stochastic Gillespie solvers isolate `type="CME"`.

**Example:**
```xml
<reaction id="ode_G2_GAE_bind" reversible="false" compartment="plasmaMembrane">
  <annotation>
    <simulation_engine xmlns="http://www.simulationobjects.com/yeastgs" type="ODE"/>
  </annotation>
</reaction>
```

---

## Part 2: RDME-ODE SBML Schema
The RDME-ODE model (`galactose_rdme_ode.xml`) directly inherits the logic established above, but heavily modifies structural metadata to enforce restrictive spatial matrices mapped out over the physical sub-cellular lattice.

### 1. Simulation Regions (Geometry Linking)
Standard SBML `<compartment>` nodes serve as the abstract hierarchy but are explicitly extended with a custom `geometry_link`. This connects the conceptual domain to an absolute file path pointing to the active internal `.npy` (NumPy) boundary matrix file required for building the 3D lattice.

**Example:**
```xml
<compartment id="plasmaMembrane" spatialDimensions="3" size="0.01" constant="true">
  <annotation>
    <geometry_link xmlns="http://www.simulationobjects.com/yeastgs">
      <file path="workspace/plasmMembrane_connected.npy"/>
    </geometry_link>
  </annotation>
</compartment>
```

### 2. Diffusion Coefficients
Every dynamic chemical species features a localized `<spatial_properties>` block embedded natively in their definition. This assigns highly selective physical Brownian bounds based on their biochemical context (e.g. tracking mobile mRNA blocks at normal speeds versus stationary `DG` chromosomal targets permanently bounded to 0.0 diffusion rules).

**Example:**
```xml
<species id="R2" compartment="nucleoplasm" initialAmount="1">
  <annotation>
    <spatial_properties xmlns="http://www.simulationobjects.com/yeastgs">
      <diffusion_coefficient value="5e-14" unit="m2/s"/>
    </spatial_properties>
  </annotation>
</species>
```

### 3. Confined Reaction Spaces
Reaction occurrences aren't uniformly ubiquitous across cellular gradients. To respect explicit physical constraints (such as forcing G1 protein translations to spawn strictly at validated ribosome domains rather than drifting universally), an explicit `reaction_space` block is deployed. 

It defines exactly which structural `<compartment>` indices are allowed to serve as valid physical reaction seeds for stochastic resolution.

**Example:**
```xml
<reaction id="transl_g1" reversible="false" compartment="cytoplasm">
  <annotation>
    <simulation_engine xmlns="http://www.simulationobjects.com/yeastgs" type="RDME">
        <reaction_space valid_compartments="cytoRibosomes,pmaRibosomes"/>
    </simulation_engine>
  </annotation>
</reaction>
```
