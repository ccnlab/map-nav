// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand"
	"github.com/goki/gi/mat32"
)

// Policy provides a parameterized default "subcortical" navigation policy
type Policy struct {
	Auto         bool       `desc:"let the network do the driving, instead of having strong training wheels"`
	UseInAct     float32    `desc:"probability of using input activation from model"`
	AvoidPGo     float32    `desc:"probability of going when avoid turning if current avg dist > prev"`
	PrvAct       Actions    `inactive:"+" desc:"previous action"`
	CurPos       mat32.Vec2 `inactive:"+" desc:"current position"`
}

func (pl *Policy) Defaults() {
	pl.UseInAct = 0.2
}

// Act is main interface call that updates dist and selects action and updates state
func (pl *Policy) Act(inact Actions) Actions {
	var act Actions
	if pl.Auto {
		act = pl.ActChooseAuto(inact)
	} else {
		act = pl.ActChooseTrain(inact)
	}
	pl.PrvAct = act
	// fmt.Printf("chose action: %v  from  in act: %v  min, avg d: %g  %g\n", act, inact, mind, avgd)
	return act
}

// ActChooseTrain makes actual choice based on current state -- called by Act -- for
// extensive training wheels mode
func (pl *Policy) ActChooseTrain(inact Actions) Actions {
	if rand.Float32() > pl.UseInAct {
	// Random Walk policy
		return Actions(rand.Intn(4))

	// walk in a circle policy
		// switch pl.PrvAct {
		// 	case North:
		// 		return East
		// 	case East:
		// 		return South
		// 	case South:
		// 		return West
		// 	case West:
		// 		return North
		// }
	}
	return inact
}

// ActChooseAuto makes actual choice based on current state -- called by Act -- for
// autonomous behavior mode -- only act when absolutely necessary
func (pl *Policy) ActChooseAuto(inact Actions) Actions {
	return inact
}
