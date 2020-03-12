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
func (pl *Policy) Act(inact Actions,ev *Env) Actions {
	var act Actions
	if pl.Auto {
		act = pl.ActChooseAuto(inact,ev )
	} else {
		act = pl.ActChooseTrain(inact, ev)
	}
	pl.PrvAct = act
	// fmt.Printf("chose action: %v  from  in act: %v  min, avg d: %g  %g\n", act, inact, mind, avgd)
	return act
}

// ActChooseTrain makes actual choice based on current state -- called by Act -- for
// extensive training wheels mode
func (pl *Policy) ActChooseTrain(inact Actions, ev *Env) Actions {
	if rand.Float32() > pl.UseInAct {
	// Random Walk policy
		// try up to 10 times to find an action that doesn't hit a wall
		for i:= 0 ; i <10 ; i++ {
			act := Actions(rand.Intn(int(ActionsN)))
			npos := Move(ev.CurPos,act)
			if ev.World.Loc(npos).open {
				return act
			}
		}

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
func (pl *Policy) ActChooseAuto(inact Actions, ev *Env) Actions {
	return inact
}
