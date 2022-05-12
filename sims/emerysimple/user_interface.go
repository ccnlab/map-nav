package main

// TODO Move to emergent/egui

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/mat32"
)

// UserInterface automatically handles creation of the GUI if requested, otherwise runs on the command line.
type UserInterface struct {
	Looper        *looper.Manager
	Network       emer.Network
	Logs          *elog.Logs
	Stats         *estats.Stats
	GUI           *egui.GUI `desc:"More directly handles graphical elements."`
	AppName       string
	AppAbout      string
	AppTitle      string
	StructForView interface{} `desc:"This might be Sim or any other object you want to display to the user."`

	// Callbacks
	InitCallback func()

	// Internal
	guiEnabled bool
}

func AddDefaultGUICallbacks(manager *looper.Manager, gui *egui.GUI) {
	for _, m := range []etime.Modes{etime.Train} {
		curMode := m // For closures.
		for _, t := range []etime.Times{etime.Trial} {
			curTime := t
			if manager.GetLoop(curMode, curTime).OnEnd.HasNameLike("UpdateNetView") {
				// There might be a case where another function also Updates the NetView, and we don't want to do it twice. In particular, Net.WtFmDWt clears some values at the end of Trial, and it wants to update the view before doing so.
				continue
			}
			manager.GetLoop(curMode, curTime).OnEnd.Add("GUI:UpdateNetView", func() {
				gui.UpdateNetView() // TODO Use update timescale variable
			})
		}
	}
}

// CreateAndRunGui creates a GUI, with which the user can control the application. It will loop forever.
func (ui *UserInterface) CreateAndRunGui() {
	ui.CreateAndRunGuiWithAdditionalConfig(func() {})
}

func (ui *UserInterface) CreateAndRunGuiWithAdditionalConfig(config func()) {
	if ui.GUI == nil {
		ui.GUI = &egui.GUI{}
	}
	ui.guiEnabled = true

	AddDefaultGUICallbacks(ui.Looper, ui.GUI)

	gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
		title := ui.AppTitle
		//cat := struct{ Name string }{Name: "Cat"}
		ui.GUI.MakeWindow(ui.StructForView, ui.AppName, title, ui.AppAbout)
		ui.GUI.CycleUpdateInterval = 10

		if ui.Network != nil {
			nv := ui.GUI.AddNetView("NetView")
			nv.Params.MaxRecs = 300
			nv.SetNet(ui.Network)
			ui.GUI.NetView.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
			ui.GUI.NetView.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
		}

		if ui.Logs != nil {
			ui.GUI.AddPlots(title, ui.Logs)
		}

		if ui.Stats != nil && len(ui.Stats.Rasters) > 0 {
			stb := ui.GUI.TabView.AddNewTab(gi.KiT_Layout, "Spike Rasters").(*gi.Layout)
			stb.Lay = gi.LayoutVert
			stb.SetStretchMax()
			for _, lnm := range ui.Stats.Rasters {
				sr := ui.Stats.F32Tensor("Raster_" + lnm)
				ui.GUI.ConfigRasterGrid(stb, lnm, sr)
			}
		}

		ui.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
			Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ui.InitCallback()
				ui.GUI.UpdateWindow()
			},
		})

		modes := []etime.Modes{}
		for m, _ := range ui.Looper.Stacks {
			modes = append(modes, m)
		} // ui.Looper.Stacks.Keys()
		ui.GUI.AddLooperCtrl(ui.Looper, modes)

		// Run custom code to configure the GUI.
		config()

		ui.GUI.FinalizeGUI(false)

		ui.GUI.Win.StartEventLoop()
	})
}

func (ui *UserInterface) AddServerButton(serverRunFunc func()) {
	// This function is only necessary if you want the network to exist in a separate thread, and you want the agent to provide a server that serves intelligent actions. It adds a button to start the server.
	ui.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Start Server", Icon: "play",
		Tooltip: "Start a server.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ui.GUI.IsRunning = true
			ui.GUI.ToolBar.UpdateActions() // Disable GUI
			go func() {
				serverRunFunc()  // The server probably runs forever.
				ui.GUI.Stopped() // Reenable GUI
			}()
		},
	})
}

func (ui *UserInterface) RunWithoutGui() {
	// TODO Something something command line here?
	ui.Looper.Run()
}

// Logging stuff

// Log is the main logging function, handles special things for different scopes
func (ui *UserInterface) log(mode etime.Modes, time etime.Times, loop looper.Loop) {
	dt := ui.Logs.Table(mode, time)
	row := dt.Rows
	if time == etime.Cycle || time == etime.Trial {
		// TODO Why?
		row = loop.Counter.Cur
	}

	ui.Logs.LogRow(mode, time, row) // also logs to file, etc
	if time == etime.Cycle {
		ui.GUI.UpdateCyclePlot(etime.Test, row)
	} else {
		ui.GUI.UpdatePlot(mode, time)
	}
}

func (ui *UserInterface) addCommonLogItems() {
	// Record time in logs
	for m, st := range ui.Looper.Stacks {
		mode := m
		for t, l := range st.Loops {
			time := t
			loop := l
			ui.Logs.AddItem(&elog.Item{
				Name: "Epoch",
				Type: etensor.INT64,
				Plot: elog.DFalse,
				Write: elog.WriteMap{
					etime.Scopes([]etime.Modes{mode}, []etime.Times{time}): func(ctx *elog.Context) {
						ctx.SetInt(loop.Counter.Cur)
					}}})
		}
	}
	ui.Logs.AddItem(&elog.Item{
		Name: "Trial",
		Type: etensor.INT64,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatInt("Trial")
			}}})
	ui.Logs.AddItem(&elog.Item{
		Name: "TrialName",
		Type: etensor.STRING,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatString("TrialName")
			}}})
	ui.Logs.AddItem(&elog.Item{
		Name: "Cycle",
		Type: etensor.INT64,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Cycle): func(ctx *elog.Context) {
				ctx.SetStatInt("Cycle")
			}}})

	// Error for output layers
	for _, olnm := range ui.Network.LayersByClass(emer.Target.String()) {
		out := ui.Network.LayerByName(olnm).(axon.AxonLayer).AsAxon()

		cosDiffMap := elog.WriteMap{}
		pctErrDiffMap := elog.WriteMap{}
		trlErrDiffMap := elog.WriteMap{}
		for m, st := range ui.Looper.Stacks {
			for iter := len(st.Order) - 1; iter >= 0; iter-- {
				i := iter // For closures
				t := st.Order[i]
				if i == len(st.Order)-1 {
					// The finest timescale, such as Cycle
					cosDiffMap[etime.Scope(m, t)] = func(ctx *elog.Context) {
						ctx.SetFloat32(out.CosDiff.Cos)
					}
					pctErrDiffMap[etime.Scope(m, t)] = func(ctx *elog.Context) {
						ctx.SetFloat64(out.PctUnitErr())
					}
					trlErrDiffMap[etime.Scope(m, t)] = func(ctx *elog.Context) {
						pcterr := out.PctUnitErr()
						trlerr := 0
						if pcterr > 0 {
							trlerr = 1
						}
						ctx.SetFloat64(float64(trlerr))
					}
				} else {
					// All other, less frequent, timescales are an aggregate
					cosDiffMap[etime.Scope(m, t)] = func(ctx *elog.Context) {
						ctx.SetAgg(ctx.Mode, st.Order[i+1], agg.AggMean)
					}
				}
			}
		}

		// Add it to the list.
		ui.Logs.AddItem(&elog.Item{
			Name:  "CosSim",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Write: cosDiffMap})
		ui.Logs.AddItem(&elog.Item{
			Name:  "PctErr",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Write: pctErrDiffMap})
		ui.Logs.AddItem(&elog.Item{
			Name:  "TrialErr",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Write: trlErrDiffMap})
	}
}

func (ui *UserInterface) addDefaultLoggingCallbacks() {
	manager := ui.Looper
	for m, loops := range manager.Stacks {
		curMode := m // For closures.
		for t, loop := range loops.Loops {
			curTime := t

			// Pass logs that haven't been configured
			_, ok := ui.Logs.Tables[etime.Scope(m, t)]
			if !ok {
				continue
			}

			// Actual logging
			loop.OnEnd.Add(curMode.String()+":"+curTime.String()+":"+"Log", func() {
				ui.log(curMode, curTime, *loop)
			})

			// Reset logs at level one deeper
			levelToReset := etime.AllTimes
			for i, tt := range loops.Order {
				if tt == t && i+1 < len(loops.Order) {
					levelToReset = loops.Order[i+1]
				}
			}
			if levelToReset != etime.AllTimes {
				loop.OnEnd.Add(curMode.String()+":"+curTime.String()+":"+"ResetLog"+levelToReset.String(), func() {
					ui.Logs.ResetLog(curMode, levelToReset)
				})
			}
		}
	}
}

func (ui *UserInterface) AddDefaultLogging() {
	if ui.Logs == nil {
		ui.Logs = &elog.Logs{}
	}
	ui.addCommonLogItems()
	ui.Logs.CreateTables() // Create them after log items have been added in addCommonLogItems
	ui.Logs.SetContext(ui.Stats, ui.Network)
	ui.addDefaultLoggingCallbacks()
}
