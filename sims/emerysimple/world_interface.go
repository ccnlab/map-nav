package main

import (
	"github.com/emer/etable/etensor"
)

// TODO Move this to a library package like emer.

// WorldInterface is like env.Env. // TODO Give better comment here.
type WorldInterface interface {
	// Init Initializes or reinitialize the world. This blocks until it hears from the world that it has been initialized. It returns the specifications for the action and observation spaces as its two return arguments.
	// The Observation Space describes the shape and names of what the model can expect as inputs. This will be constant across the run.
	// The Action Space describes the shape and names of what the model can send as outputs. This will be constant across the run.
	Init(details string) (map[string]SpaceSpec, map[string]SpaceSpec)

	// Step the environment. It takes in a set of actions and returns observations and a debug string.
	// The actions should conform to the action space specification.
	// The observations can be expected to conform to the observation space specification. The observations will be cached such that a separate function can get them before the next time Step is called.
	// The debug string should not be used for actual training.
	Step(actions map[string]Action, agentDone bool) (map[string]etensor.Tensor, bool, string)

	// Observe Returns a tensor for the named modality. E.g. “x” or “vision” or “reward”. This just returns a cached entry into the map gotten the last time Step was called.
	Observe(name string) etensor.Tensor

	// ObserveWithShape Returns a tensor for the named modality like Observe. This allows the agent to request an observation in a specific shape, which may involve downsampling. It should throw an error if the shap can't be satisfied.
	ObserveWithShape(name string, shape SpaceSpec) etensor.Tensor
}
