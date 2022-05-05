package main

import (
	"fmt"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/patgen"
	"github.com/emer/etable/etensor"
)

// Proof of concept for replacing the environment with a simpler environment that uses the network.

type ExampleWorld struct {
	WorldInterface
	Nm  string `desc:"name of this environment"`
	Dsc string `desc:"description of this environment"`

	// Counters
	Run   env.Ctr `view:"inline" desc:"current run of model as provided during Init"`
	Epoch env.Ctr `view:"inline" desc:"increments over arbitrary fixed number of trials, for general stats-tracking"`
	Trial env.Ctr `view:"inline" desc:"increments for each step of world, loops over epochs -- for general stats-tracking independent of env state"`

	observationShape map[string][]int
	observations     map[string]*etensor.Float32
}

//Display displays the environmnet
func (world *ExampleWorld) Display() {
	fmt.Println("This world state")
}

// Init Initializes or reinitialize the world, todo, change from being hardcoded for emery
func (world *ExampleWorld) Init(details string) {
	fmt.Println("Init Dworld: " + details)

	world.observationShape = make(map[string][]int)
	world.observations = make(map[string]*etensor.Float32)

	world.observationShape["VL"] = []int{5, 5}
	world.observationShape["Act"] = []int{5, 5}

	world.observations["VL"] = etensor.NewFloat32(world.observationShape["VL"], nil, nil)
	world.observations["Act"] = etensor.NewFloat32(world.observationShape["Act"], nil, nil)

	patgen.PermutedBinaryRows(world.observations["VL"], 1, 1, 0)
	patgen.PermutedBinaryRows(world.observations["Act"], 3, 1, 0)

}

func (world *ExampleWorld) GetObservationSpace() map[string][]int {
	return world.observationShape //todo need to instantitate
}

func (world *ExampleWorld) GetActionSpace() map[string][]int {
	return nil
}

func (world *ExampleWorld) DecodeAndTakeAction(action string, vt *etensor.Float32) string {
	fmt.Printf("Prediciton of the network")
	fmt.Printf(vt.String())
	fmt.Println("\n Expected Output")
	fmt.Printf(world.observations["VL"].String())
	return "Taking in info from model and moving forward"
}

// StepN Updates n timesteps (e.g. milliseconds)
func (world *ExampleWorld) StepN(n int) {}

// Step 1
func (world *ExampleWorld) Step() {
	fmt.Println("I'm taking a step")

}

// Observe Returns a tensor for the named modality. E.g. “x” or “vision” or “reward”
func (world *ExampleWorld) Observe(name string) etensor.Tensor {
	//constantly rnadomize this input, i know it says Act, but it is treated as an input, confusing?, need to V2wdp, etc
	//to do, make less sparse input
	world.observations["Act"] = etensor.NewFloat32(world.observationShape["Act"], nil, nil)
	return world.observations[name]
}

func (world *ExampleWorld) ObserveWithShape(name string, shape []int) etensor.Tensor {
	return world.observations[name]
}

// Action Output action to the world with details. Details might contain a number or array. So this might be Action(“move”, “left”) or Action(“LeftElbow”, “0.4”) or Action("Log", "[0.1, 0.9]")
func (world *ExampleWorld) Action(action, details string) {}

// Done Returns true if episode has ended, e.g. when exiting maze
func (world *ExampleWorld) Done() bool {
	return false
}

// Info Returns general information about the world, for debugging purposes. Should not be used for actual learning.
func (world *ExampleWorld) Info() string {
	return "God is dead"
}

func (world *ExampleWorld) GetCounter(time etime.Times) env.Ctr {
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
