package main

// TODO Move to a different package in emer

import "github.com/emer/etable/etensor"

// The Shape of an action or observation.
// It should *either* be continuous with a Shape or discrete, but not both. If both are set, treat it as continuous, using Shape.
type Shape struct {
	// Continuous
	ContinuousShape []int `desc:"The dimensions of an array. For example, [3,2] would be a 3 by 2 array. [1] would be a single value."`
	Stride          []int // TODO Describe
	Min             float `desc:"The minimum continuous value."`
	Max             float `desc:"The maximum continuous value."`

	// Discrete
	DiscreteLabels []string `desc:"The names of the discrete possibilities, such as ['left', 'right']. The length of this is the number of discrete possibilities that this shape encapsulates."`
}

// An Action describes what the agent is doing at a given timestep. It should contain either a continuous vector or a discrete option, as specified by its shape.
type Action struct {
	ActionShape    *Shape         `desc:"Optional description of the action."`
	Vector         etensor.Tensor `desc:"A vector describing the action. For example, this might be joint positions or forces applied to actuators."`
	DiscreteOption int            `desc:"Choice from among the DiscreteLabels in Continuous."`
}

// AgentInterface allows the Agent to provide actions given observations. This allows the agent to be embedded within a world.
type AgentInterface interface {
	// Init passes variables to the Agent: Action space, Observation space, and initial Observation. It receives any specification in the form of a string which the agent chooses to provide. Agent should reinitialize the network for the beginning of a new run.
	Init(actionSpace map[string]Shape, observationSpace map[string]Shape) string

	// Step takes in a map of named Observations. It returns a map of named Actions. The observation can be expected to conform to the shape given in Init, and the Action should conform to the action specification given there. The debug string is for debug information from the environment and should not be used for real training or evaluation.
	Step(observations map[string]etensor.Tensor, debug string) map[string]Action
}
