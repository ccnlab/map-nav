package main

// TODO This can be deleted.

import (
	"fmt"
	"github.com/emer/emergent/patgen"
	"github.com/emer/etable/etensor"
)

// Proof of concept for replacing the environment with a simpler environment that uses the network.

type ExampleWorld struct {
	WorldInterface
	Nm  string `desc:"name of this environment"`
	Dsc string `desc:"description of this environment"`

	observationShape map[string][]int
	observations     map[string]*etensor.Float32
}

//Display displays the environmnet
func (world *ExampleWorld) Display() {
	//fmt.Println("This world state")
	fmt.Println("Display")
}

// Init Initializes or reinitialize the world, todo, change from being hardcoded for emery
func (world *ExampleWorld) Init(details string) {
	fmt.Println("Init Example World: " + details)

	world.observationShape = make(map[string][]int)
	world.observations = make(map[string]*etensor.Float32)

	world.observationShape["VL"] = []int{5, 5}
	world.observationShape["Act"] = []int{5, 5}

	world.observations["VL"] = etensor.NewFloat32(world.observationShape["VL"], nil, nil)
	//world.observations["Act"] = etensor.NewFloat32(world.observationShape["Act"], nil, nil)

	patgen.PermutedBinaryRows(world.observations["VL"], 1, 1, 0)
	//patgen.PermutedBinaryRows(world.observations["Act"], 3, 1, 0)

}

func (world *ExampleWorld) GetObservationSpace() map[string][]int {
	fmt.Println("GetObservationSpace")
	return world.observationShape //todo need to instantitate
}

func (world *ExampleWorld) GetActionSpace() map[string][]int {
	fmt.Println("GetActionSpace")
	return nil
}

func (world *ExampleWorld) DecodeAndTakeAction(action string, vt *etensor.Float32) string {
	//fmt.Printf("Prediciton of the network")
	//fmt.Printf(vt.String())
	//fmt.Println("\n Expected Output")
	//fmt.Printf(world.observations["VL"].String())
	fmt.Println("DecodeAndTakeAction")
	return "Taking in info from model and moving forward"
}

// StepN Updates n timesteps (e.g. milliseconds)
func (world *ExampleWorld) StepN(n int) {
	fmt.Println("StepN")
}

// Step 1
func (world *ExampleWorld) Step() {
	//fmt.Println("I'm taking a step")
	fmt.Println("Step")
}

// Observe Returns a tensor for the named modality. E.g. “x” or “vision” or “reward”
func (world *ExampleWorld) Observe(name string) etensor.Tensor {
	fmt.Println("Observe:" + name)
	//constantly rnadomize this input, i know it says Act, but it is treated as an input, confusing?, need to V2wdp, etc
	//to do, make less sparse input
	world.observations["Act"] = etensor.NewFloat32(world.observationShape["Act"], nil, nil)
	return world.observations[name]
}

func (world *ExampleWorld) ObserveWithShape(name string, shape []int) etensor.Tensor {
	fmt.Println("ObserveWithShape:" + name)
	return world.observations[name]
}

func (world *ExampleWorld) ObserveWithShapeStride(name string, shape []int, strides []int) etensor.Tensor {
	fmt.Println("ObserveWithShapeStride:" + name)
	if name == "VL" { //if type target
		return world.observations[name]
	} else { //if an input
		if world.observations[name] == nil {
			//fmt.Println("CALLED ONCE")
			world.observations[name] = etensor.NewFloat32(shape, strides, nil)
			patgen.PermutedBinaryRows(world.observations[name], 1, 1, 0)
		}
		return world.observations[name]
	}
}

// Action Output action to the world with details. Details might contain a number or array. So this might be Action(“move”, “left”) or Action(“LeftElbow”, “0.4”) or Action("Log", "[0.1, 0.9]")
func (world *ExampleWorld) Action(action, details string) {
	fmt.Println("Action:" + action)
}

// Done Returns true if episode has ended, e.g. when exiting maze
func (world *ExampleWorld) Done() bool {
	fmt.Println("Done")
	return false
}

// Info Returns general information about the world, for debugging purposes. Should not be used for actual learning.
func (world *ExampleWorld) Info() string {
	fmt.Println("Info")
	return "God is dead"
}
