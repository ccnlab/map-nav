package main

import (
	"fmt"
	"github.com/emer/axon/axon"
	"time"
)

// LogPrec is precision for saving float values in logs
const LogPrec = 4 // TODO(refactor): Logs library

func ToggleLayersOff(net *axon.Network, layerNames []string, off bool) { // TODO(refactor): move to library
	for _, lnm := range layerNames {
		lyi := net.LayerByName(lnm)
		if lyi == nil {
			fmt.Printf("layer not found: %s\n", lnm)
			continue
		}
		lyi.SetOff(off)
	}
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func NewRndSeed(randomSeed *int64) { // TODO(refactor): to library
	*randomSeed = time.Now().UnixNano()
}
