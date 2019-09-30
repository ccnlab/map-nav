// Copyright (c) 2019, The CCNLab Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package navenv

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/epe/epe"
	"github.com/emer/etable/etensor"
)

// Env manages the navigation environment
type Env struct {
	Nm           string     `desc:"name of this environment"`
	Dsc          string     `desc:"description of this environment"`
	World        *epe.Group `desc:"physics engine world"`
	Run          env.Ctr    `view:"inline" desc:"current run of model as provided during Init"`
	Epoch        env.Ctr    `view:"inline" desc:"number of times through entire set of patterns"`
	Trial        env.Ctr    `view:"inline" desc:"current ordinal item in Table -- if Sequential then = row number in table, otherwise is index in Order list that then gives row number in Table"`
	TrialName    string     `desc:"if Table has a Name column, this is the contents of that for current trial"`
	PrvTrialName string     `desc:"previous trial name"`
}

func (ft *Env) Name() string { return ft.Nm }
func (ft *Env) Desc() string { return ft.Dsc }

func (ft *Env) Validate() error {
	ft.Run.Scale = Run
	ft.Epoch.Scale = Epoch
	ft.Trial.Scale = Trial
	if ft.Table == nil || ft.Table.Table == nil {
		return fmt.Errorf("env.Env: %v has no Table set", ft.Nm)
	}
	if ft.Table.Table.NumCols() == 0 {
		return fmt.Errorf("env.Env: %v Table has no columns -- Outputs will be invalid", ft.Nm)
	}
	return nil
}

func (ft *Env) Counters() []TimeScales {
	return []TimeScales{Run, Epoch, Trial}
}

func (ft *Env) States() Elements {
	els := Elements{}
	els.FromSchema(ft.Table.Table.Schema())
	return els
}

func (ft *Env) Actions() Elements {
	return nil
}

func (ft *Env) Init(run int) {
	if ft.World == nil {
		ft.World = &epe.Group{}
		ft.World.InitName(ft.World, "World")
	}
	ft.Run.Init()
	ft.Epoch.Init()
	ft.Trial.Init()
	ft.Run.Cur = run
	np := ft.Table.Len()
	ft.Order = rand.Perm(np) // always start with new one so random order is identical
	// and always maintain Order so random number usage is same regardless, and if
	// user switches between Sequential and random at any point, it all works..
	ft.Trial.Max = np
	ft.Trial.Cur = -1 // init state -- key so that first Step() = 0
}

// Row returns the current row number in table, based on Sequential / perumuted Order and
// already de-referenced through the IdxView's indexes to get the actual row in the table.
func (ft *Env) Row() int {
	if ft.Sequential {
		return ft.Table.Idxs[ft.Trial.Cur]
	}
	return ft.Table.Idxs[ft.Order[ft.Trial.Cur]]
}

func (ft *Env) SetTrialName() {
	if nms := ft.Table.Table.ColByName("Name"); nms != nil {
		ft.TrialName = nms.StringVal1D(ft.Row())
	}
}

func (ft *Env) Step() bool {
	ft.Epoch.Same() // good idea to just reset all non-inner-most counters at start

	if ft.Trial.Incr() { // if true, hit max, reset to 0
		erand.PermuteInts(ft.Order)
		ft.Epoch.Incr()
	}
	ft.PrvTrialName = ft.TrialName
	ft.SetTrialName()
	return true
}

func (ft *Env) State(element string) etensor.Tensor {
	et, err := ft.Table.Table.CellTensorTry(element, ft.Row())
	if err != nil {
		log.Println(err)
	}
	return et
}

func (ft *Env) Action(element string, input etensor.Tensor) {
	// nop
}

func (ft *Env) Counter(scale TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case Run:
		return ft.Run.Query()
	case Epoch:
		return ft.Epoch.Query()
	case Trial:
		return ft.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ Env = (*env.Env)(nil)
