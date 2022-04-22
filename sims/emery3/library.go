package main

import (
	"fmt"
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/empi/mpi"
	"log"
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

////////////////////////////////////////////////////////////////////
//  MPI code

// MPIInit initializes MPI
func MPIInit(useMPI *bool) *mpi.Comm { // TODO(refactor): library code
	mpi.Init()
	var err error

	comm, err := mpi.NewComm(nil) // use all procs
	if err != nil {
		log.Println(err)
		*useMPI = false
	} else {
		mpi.Printf("MPI running on %d procs\n", mpi.WorldSize())
	}
	return comm
}

// MPIFinalize finalizes MPI
func MPIFinalize(useMPI bool) { // TODO(refactor): library code
	if useMPI {
		mpi.Finalize()
	}
}

// CollectDWts collects the weight changes from all synapses into AllDWts
func CollectDWts(net *axon.Network, allWeightChanges *[]float32) { // TODO(refactor): axon library code
	net.CollectDWts(allWeightChanges)
}

// MPIWtFmDWt updates weights from weight changes, using MPI to integrate
// DWt changes across parallel nodes, each of which are learning on different
// sequences of inputs.
func MPIWtFmDWt(comm *mpi.Comm, net *deep.Network, useMPI bool, allWeightChanges *[]float32, sumWeights *[]float32, time axon.Time) { // TODO(refactor): axon library code

	if useMPI {
		CollectDWts(&net.Network, allWeightChanges)
		ndw := len(*allWeightChanges)
		if len(*sumWeights) != ndw {
			*sumWeights = make([]float32, ndw)
		}
		comm.AllReduceF32(mpi.OpSum, *sumWeights, *allWeightChanges)
		net.SetDWts(*sumWeights, mpi.WorldSize())
	}
	net.WtFmDWt(&time)
}
