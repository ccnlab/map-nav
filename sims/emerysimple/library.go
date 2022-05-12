package main

import (
	"fmt"
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/relpos"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gi"
	"github.com/goki/mat32"
	"log"
	"math"
	"math/rand"
	"time"
)

// TODO This library should be broken up and sent to different places.

// LogPrec is precision for saving float values in logs
const LogPrec = 4 // TODO(refactor): Logs library

func (ss *Sim) AddDefaultLoopSimLogic(manager *looper.Manager) {
	// Net Cycle
	for m, _ := range manager.Stacks {
		manager.Stacks[m].Loops[etime.Cycle].Main.Add("Axon:Cycle:RunAndIncrement", func() {
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
		})
	}
	// Weight updates.
	// Note that the substring "UpdateNetView" in the name is important here, because it's checked in AddDefaultGUICallbacks.
	manager.GetLoop(etime.Train, etime.Trial).OnEnd.Add("Axon:LoopSegment:UpdateWeights", func() {
		ss.Net.DWt(&ss.Time)
		// TODO Need to update net view here to accurately display weight changes.
		ss.Net.WtFmDWt(&ss.Time)
	})

	// Set variables on ss that are referenced elsewhere, such as ApplyInputs.
	for m, loops := range manager.Stacks {
		curMode := m // For closures.
		for t, loop := range loops.Loops {
			curTime := t
			loop.OnStart.Add(curMode.String()+":"+curTime.String()+":"+"SetTimeVal", func() {
				ss.Time.Mode = curMode.String()
			})
		}
	}
}

// SendActionAndStep takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
func (ss *Sim) SendActionAndStep(net *deep.Network, ev WorldInterface) {
	// Get the first Target (output) layer
	actions := map[string]Action{}
	for _, lnm := range net.LayersByClass(emer.Target.String()) {
		ly := net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		vt := &etensor.Float32{}
		ly.UnitValsTensor(vt, "ActM")
		actions[lnm] = Action{Vector: vt}
	}
	_, _, debug := ev.Step(actions, false)
	if debug != "" {
		fmt.Println("Got debug from Step: " + debug)
	}
}

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

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func ApplyInputs(net *deep.Network, en WorldInterface, states, layers []string) { // TODO(refactor): library code
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	for i, lnm := range layers {
		lyi := net.LayerByName(lnm)
		if lyi == nil {
			continue
		}
		ly := lyi.(axon.AxonLayer).AsAxon()
		ss := SpaceSpec{ContinuousShape: lyi.Shape().Shp, Stride: lyi.Shape().Strd}
		pats := en.ObserveWithShape(states[i], ss)
		//lyi.Shape().Strides()
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

func ApplyInputsWithStrideAndShape(net *deep.Network, en WorldInterface, states, layers []string) {
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	for i, lnm := range layers {
		lyi := net.LayerByName(lnm)
		if lyi == nil {
			continue
		}
		ly := lyi.(axon.AxonLayer).AsAxon()
		ss := SpaceSpec{ContinuousShape: lyi.Shape().Shp, Stride: lyi.Shape().Strd}
		pats := en.ObserveWithShape(states[i], ss)
		lyi.Shape().Strides()
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func SaveWeights(fileName string, net *deep.Network) { // TODO(refactor): library code
	fnm := fileName
	fmt.Printf("Saving Weights to: %v\n", fnm)
	net.SaveWtsJSON(gi.FileName(fnm))
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

/////////////////////////////////////////////////////////////////////////
//   Params setting

// ParamsName returns name of current set of parameters
func ParamsName(paramset string) string { // TODO(refactor): library code
	if paramset == "" {
		return "Base"
	}
	return paramset
}

// TODO(refactor): Fix "Network" and "Sim" as arguments below

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func SetParams(sheet string, setMsg bool, net *deep.Network, params *params.Sets, paramName string, ss interface{}) error { // TODO(refactor): Move to library, take in names as args
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := SetParamsSet("Base", sheet, setMsg, net, params, ss)
	if paramName != "" && paramName != "Base" {
		err = SetParamsSet(paramName, sheet, setMsg, net, params, ss)
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func SetParamsSet(setNm string, sheet string, setMsg bool, net *deep.Network, params *params.Sets, ss interface{}) error { // TODO(refactor): library, take in names as args
	pset, err := params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func RunName(tag string, paramName string) string { // TODO(refactor): library code
	if tag != "" {
		return tag + "_" + ParamsName(paramName)
	} else {
		return ParamsName(paramName)
	}
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func RunEpochName(run, epc int) string { // TODO(refactor): library
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func WeightsFileName(netName, tag, paramName string, run, epc int) string { // TODO(refactor): library
	return netName + "_" + RunName(tag, paramName) + "_" + RunEpochName(run, epc) + ".wts.gz"
}

// LogFileName returns default log file name
func LogFileName(netName, lognm, tag, paramName string) string { // TODO(refactor): library
	return netName + "_" + RunName(tag, paramName) + "_" + lognm + ".tsv"
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// ValsTsr gets value tensor of given name, creating if not yet made
func ValsTsr(tensorDictionary *map[string]*etensor.Float32, name string) *etensor.Float32 { // TODO(refactor): library code

	if *tensorDictionary == nil {
		*tensorDictionary = make(map[string]*etensor.Float32)
	}
	tsr, ok := (*tensorDictionary)[name]
	if !ok {
		tsr = &etensor.Float32{}
		(*tensorDictionary)[name] = tsr
	}
	return tsr
}

// HogDead computes the proportion of units in given layer name with ActAvg over hog thr
// and under dead threshold
func HogDead(net *deep.Network, lnm string) (hog, dead float64) { // TODO(refactor): library stats code
	ly := net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
	n := len(ly.Neurons)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.ActAvg > 0.3 {
			hog += 1
		} else if nrn.ActAvg < 0.01 {
			dead += 1
		}
	}
	hog /= float64(n)
	dead /= float64(n)
	return
}

// AddPlusAndMinusPhases adds the minus and plus phases of the theta cycle, which help the network learn.
func AddPlusAndMinusPhases(manager *looper.Manager, time *axon.Time, net *deep.Network) (looper.Event, looper.Event, looper.Event) {
	// The minus and plus phases of the theta cycle, which help the network learn.
	minusPhase := looper.Event{Name: "MinusPhase", AtCtr: 0}
	minusPhase.OnEvent.Add("Sim:MinusPhase:Start", func() {
		time.PlusPhase = false
		time.NewPhase(false)
	})
	plusPhase := looper.Event{Name: "PlusPhase", AtCtr: 150}
	plusPhase.OnEvent.Add("Sim:MinusPhase:End", func() { net.MinusPhase(time) })
	plusPhase.OnEvent.Add("Sim:PlusPhase:Start", func() {
		time.PlusPhase = true
		time.NewPhase(true)
	})
	plusPhaseEnd := looper.Event{Name: "PlusPhase", AtCtr: 199}
	plusPhaseEnd.OnEvent.Add("Sim:PlusPhase:End", func() { net.PlusPhase(time) })
	// Add both to train and test, by copy
	manager.AddEventAllModes(etime.Cycle, minusPhase)
	manager.AddEventAllModes(etime.Cycle, plusPhase)
	manager.AddEventAllModes(etime.Cycle, plusPhaseEnd)
	return minusPhase, plusPhase, plusPhaseEnd
}

// TODO All of the following is for automatically placing layers. It works OK but it's probably overcomplicated.
func computeLayerOverlap(lay1 emer.Layer, lay2 emer.Layer) float32 {
	s1 := lay1.Size()
	s2 := lay2.Size()
	// Overlap in X
	xo := float32(0)
	xlow := math.Max(float64(lay1.Pos().X), float64(lay2.Pos().X))
	xhigh := math.Max(float64(lay1.Pos().X+s1.X), float64(lay2.Pos().X+s2.X))
	if xhigh > xlow {
		xo = float32(xhigh - xlow)
	}
	// Overlap in Z
	zo := float32(0)
	zlow := math.Max(float64(lay1.Pos().Z), float64(lay2.Pos().Z))
	zhigh := math.Max(float64(lay1.Pos().Z+s1.Y), float64(lay2.Pos().Z+s2.Y)) * 2
	if zhigh > zlow {
		zo = float32(zhigh - zlow)
	}
	// Overlap is the product of overlap in both dimensions.
	return xo * zo
}

func scoreNet(net emer.Network, pctDone float32) float32 {
	score := float32(0)
	idealDist := float32(5)
	idealDistUnconnect := float32(15)
	unconnectedTerm := float32(0.1)
	inOutPosTerm := float32(100)
	negativeTerm := float32(10000)                 // This should be much bigger than inOutPosTerm
	overlapTerm := float32(10) * pctDone * pctDone // Care a lot more about overlap as we go
	for i := 0; i < net.NLayers(); i++ {
		layer := net.Layer(i)
		// Connected layers about the right distance apart.
		connectedLayers := map[emer.Layer]bool{}
		for j := 0; j < net.Layer(i).NSendPrjns(); j++ {
			recLayer := layer.SendPrjn(j).RecvLay()
			connectedLayers[recLayer] = true
			pos1 := layer.Pos()
			pos2 := recLayer.Pos()
			dist := mat32.Sqrt((pos1.X-pos2.X)*(pos1.X-pos2.X) + (pos1.Y-pos2.Y)*(pos1.Y-pos2.Y))
			score += (dist - idealDist) * (dist - idealDist)
		}
		// Other layers a good distance away too
		for j := 0; j < net.NLayers(); j++ {
			_, ok := connectedLayers[net.Layer(j)]
			if !ok {
				pos1 := layer.Pos()
				pos2 := net.Layer(j).Pos()
				dist := mat32.Sqrt((pos1.X-pos2.X)*(pos1.X-pos2.X) + (pos1.Y-pos2.Y)*(pos1.Y-pos2.Y))
				score += (dist - idealDistUnconnect) * (dist - idealDistUnconnect) * unconnectedTerm
			}
		}
		// No overlap.
		for j := 0; j < net.NLayers(); j++ {
			if i != j {
				score -= computeLayerOverlap(layer, net.Layer(j)) * overlapTerm
			}
		}
		// Inputs to the bottom, outputs to the top.
		if layer.Type() == emer.Input {
			score -= layer.Pos().Z * inOutPosTerm
		}
		if layer.Type() == emer.Target {
			score += layer.Pos().Z * inOutPosTerm
		}
		// Don't go negative.
		if layer.Pos().Z < 0 {
			score += layer.Pos().Z * negativeTerm
		}
		//if layer.Pos().X < 0 {
		//	score += layer.Pos().X * negativeTerm
		//}
	}
	return score
}

// PositionNetworkLayersAutomatically tries to find a configuration for the network layers where they're close together, but not overlapping. It tries to put connected layers closer together, input layers near the bottom, and target layers near the top. It uses a random walk algorithm that randomly permutes the network and only keeps permutations if they improve the network's overall configuration score.
// numSettlingIterations is the number of random moves it tries for each layer. Larger values will generally get better results but compute time grows linearly.
func PositionNetworkLayersAutomatically(net emer.Network, numSettlingIterations int) {
	size := float32(50) // The size of the positioning area
	wiggleSize := float32(5)
	// Initially randomize layers
	for j := 0; j < net.NLayers(); j++ {
		layer := net.Layer(int(j))
		layer.SetRelPos(relpos.Rel{Rel: relpos.NoRel})
		layer.SetPos(mat32.Vec3{rand.Float32() * size, 0, rand.Float32() * size})
	}
	for i := 0; i < numSettlingIterations; i++ {
		for j := 0; j < net.NLayers(); j++ {
			layer := net.Layer(int(j))
			pos := layer.Pos()
			// Make a random change and see if it improves things.
			offset := mat32.Vec3{rand.Float32()*wiggleSize - wiggleSize/2, 0, rand.Float32()*wiggleSize - wiggleSize/2}
			beforeScore := scoreNet(net, float32(i)/float32(numSettlingIterations))
			newPos := mat32.Vec3{pos.X + offset.X, pos.Y + offset.Y, pos.Z + offset.Z}
			layer.SetPos(newPos)
			afterScore := scoreNet(net, float32(i)/float32(numSettlingIterations))
			if beforeScore > afterScore {
				// Revert this random change.
				layer.SetPos(pos)
			}
		}
		// Simulated annealing.
		wiggleSize = wiggleSize * (1 - 1/float32(numSettlingIterations))
	}
}
