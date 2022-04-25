package main

import "github.com/emer/etable/etensor"

// Proof of concept for replacing the environment with a simpler environment that uses the network.

type DWorld struct {
	Nm  string `desc:"name of this environment"`
	Dsc string `desc:"description of this environment"`
}

// Init Initializes or reinitialize the world
func (world *DWorld) Init(details string) {}

// StepN Updates n timesteps (e.g. milliseconds)
func (world *DWorld) StepN(n int) {}

// Step 1
func (world *DWorld) Step() {}

// Observe Returns a tensor for the named modality. E.g. “x” or “vision” or “reward”
func (world *DWorld) Observe(name string) etensor.Tensor {
	return nil
}

// Action Output action to the world with details. Details might contain a number or array. So this might be Action(“move”, “left”) or Action(“LeftElbow”, “0.4”) or Action("Log", "[0.1, 0.9]")
func (world *DWorld) Action(action, details string) {}

// Done Returns true if episode has ended, e.g. when exiting maze
func (world *DWorld) Done() bool {
	return false
}

// Info Returns general information about the world, for debugging purposes. Should not be used for actual learning.
func (world *DWorld) Info() string {
	return "God is dead"
}
