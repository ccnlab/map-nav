package main

import (
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/etime"
	"github.com/emer/etable/etensor"
)

// TODO Move this to a library package like emer.

// WorldInterface is like env.Env.
type WorldInterface interface {
	// Init Initializes or reinitialize the world
	Init(details string)

	// StepN Updates n timesteps (e.g. milliseconds)
	StepN(n int)

	// Step 1
	Step()

	// GetObservationSpace describes the shape and names of what the model can expect as inputs
	GetObservationSpace() map[string][]int

	// GetActionSpace describes the shape and names of what the model can send as outputs
	GetActionSpace() map[string][]int

	// Observe Returns a tensor for the named modality. E.g. “x” or “vision” or “reward”
	Observe(name string) etensor.Tensor

	// ObserveWithShape Returns a tensor for the named modality. E.g. “x” or “vision” or “reward” but returns a specific shape, like having four eyes versus 2 eyes
	ObserveWithShape(name string, shape []int) etensor.Tensor

	// Action Output action to the world with details. Details might contain a number or array. So this might be Action(“move”, “left”) or Action(“LeftElbow”, “0.4”) or Action("Log", "[0.1, 0.9]")
	Action(action, details string)

	// Done Returns true if episode has ended, e.g. when exiting maze
	Done() bool

	// Info Returns general information about the world, for debugging purposes. Should not be used for actual learning.
	Info() string

	GetCounter(time etime.Times) env.Ctr

	DecodeAndTakeAction(vt *etensor.Float32) string

	// Display displays environment to the user
	Display()
}
