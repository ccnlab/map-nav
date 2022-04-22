package main

import (
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/etime"
	"github.com/emer/etable/etensor"
)

// Proof of concept for replacing the environment with a simpler environment that uses the network.

// WorldInterface is like env.Env.
type WorldInterface interface {
	// Init Initializes or reinitialize the world
	Init(details string)

	// StepN Updates n timesteps (e.g. milliseconds)
	StepN(n int)

	// Step 1
	Step()

	// Observe Returns a tensor for the named modality. E.g. “x” or “vision” or “reward”
	Observe(name string) etensor.Tensor

	// Action Output action to the world with details. Details might contain a number or array. So this might be Action(“move”, “left”) or Action(“LeftElbow”, “0.4”) or Action("Log", "[0.1, 0.9]")
	Action(action, details string)

	// Done Returns true if episode has ended, e.g. when exiting maze
	Done() bool

	// Info Returns general information about the world, for debugging purposes. Should not be used for actual learning.
	Info() string

	GetCounter(time etime.Times) env.Ctr
}

type DWorld struct {
	WorldInterface
	Name string `desc:"name of this environment"`
	Desc string `desc:"description of this environment"`

	Run   env.Ctr `view:"inline" desc:"current run of model as provided during Init"`
	Epoch env.Ctr `view:"inline" desc:"increments over arbitrary fixed number of trials, for general stats-tracking"`
	Trial env.Ctr `view:"inline" desc:"increments for each step of world, loops over epochs -- for general stats-tracking independent of env state"`
}

// Init Initializes or reinitialize the world
func (world *DWorld) Init(details string) {
	//socketlibrary.Init(details)
}

// StepN Updates n timesteps (e.g. milliseconds)
func (world *DWorld) StepN(n int) {}

// Step 1
func (world *DWorld) Step() {}

// Observe Returns a tensor for the named modality. E.g. “x” or “vision” or “reward”
func (world *DWorld) Observe(name string) etensor.Tensor {
	return nil
}

// Action Output action to the world with details. Details might contain a number or array. So this might be Action(“move”, “left”) or Action(“LeftElbow”, “0.4”) or Action("Log", "[0.1, 0.9]")
func (world *DWorld) Action(action, details string) {
	print("Got this action! " + action + ", " + details)
}

// Done Returns true if episode has ended, e.g. when exiting maze
func (world *DWorld) Done() bool {
	return false
}

// Info Returns general information about the world, for debugging purposes. Should not be used for actual learning.
func (world *DWorld) Info() string {
	return "God is dead"
}

func (world *DWorld) GetCounter(time etime.Times) env.Ctr {
	if time == etime.Epoch {
		return world.Epoch
	}
	if time == etime.Trial {
		return world.Trial
	}
	if time == etime.Run {
		return world.Run
	}
	print("You passed in a bad time request :(")
	return world.Trial
}
