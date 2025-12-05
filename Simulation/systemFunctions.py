# import os, time, subprocess
# import comtypes.client
# import numpy as np
#
#
# class DistillationColumnAspen:
#     def __init__(self, dyn_file,
#                  ss_inputs, initialization_point,
#                  ad_exe=r"C:\Program Files\AspenTech\AMSystem V14.0\Bin\AspenModeler.exe",
#                  visible=True):
#
#         # Open Aspen Modeler and the dynamics File
#         dyn_file = os.path.abspath(dyn_file)
#         if not os.path.isfile(dyn_file):
#             raise FileNotFoundError(dyn_file)
#
#         # start the requested version in its own process
#         subprocess.Popen([ad_exe, dyn_file])
#
#         # Let's wait for the application to open for 20 seconds
#         time.sleep(20)
#
#         prog_id = "AD Application"
#         self.ad = comtypes.client.GetActiveObject(prog_id)
#
#         self.ad.Visible = bool(visible)
#
#         self.sim = self.ad.Simulation
#         self.fsheet = self.sim.Flowsheet
#         self.streams = self.fsheet.Streams
#         self.block = self.fsheet.Blocks
#         self.col = self.block.Item("C2S")
#
#         self.initialization_point = initialization_point
#         self.ss_inputs = ss_inputs
#
#         self.y_ss = self.ss_outputs()
#         self.current_input = self.ss_inputs.copy()
#         self.current_output = self.y_ss.copy()
#
#     def enable_records(self):
#         # Enabling records
#         self.col.Reflux.FmR.Record = True
#         self.col.QRebR.Record = True
#         # For C2H6
#         self.col.Stage[24].y[1].Record = True
#         self.col.Stage[85].T.Record = True
#         print("Records have been enabled")
#
#     def initialize_system(self):
#
#         # Set initialization values
#         feed = self.streams.Item("Feed")
#         feed.FmR.Value = self.initialization_point[0]
#         feed.T.Value = self.initialization_point[1]
#         feed.P.Value = self.initialization_point[2]
#         # C2H4
#         feed.ZR[0].Value = self.initialization_point[3]
#         # C2H6
#         feed.ZR[1].Value = self.initialization_point[4]
#
#         hx = self.block.Item("HX")
#         hx.T.Spec = "Fixed"
#         hx.T.Value = self.initialization_point[5]
#         hx.QR.Spec = "Free"
#
#         # Run Initialization
#         # Control Values all to auto
#         self.fsheet.TC.Cascade.Value = 0
#         self.fsheet.TC.AutoMan.Value = 0
#         self.fsheet.EAC.Cascade.Value = 0
#         self.fsheet.EAC.AutoMan.Value = 0
#
#         self.sim.RunMode = "Initialization"
#         self.sim.Run(True)
#         if self.sim.Successful:
#             print("Initialization has been completed")
#         else:
#             print("Initialization has been failed")
#
#         self.fsheet.TC.Cascade.Value = 1
#         self.fsheet.TC.AutoMan.Value = 0
#         self.fsheet.EAC.Cascade.Value = 1
#         self.fsheet.EAC.AutoMan.Value = 0
#
#         # Run Steady States (This is a step in Aspen dynamics. It doesn't mean steady state conditions)
#         self.sim.RunMode = "Steady State"
#         self.sim.Run(True)
#         if self.sim.Successful:
#             print("Steady State has been completed")
#         else:
#             print("Steady State has been failed")
#
#         # Now make the system Open Loop
#         for block_name in ["TC", "EAC"]:
#             self.fsheet.RemoveBlock(block_name)
#         for stream_name in ["S1", "S2", "S4", "S9"]:
#             self.fsheet.RemoveStream(stream_name)
#
#         print("System is in open loop condition now!")
#
#         self.col.Reflux.FmR.Value = self.ss_inputs[0]
#         self.col.QRebR.Value = self.ss_inputs[1]
#
#     def ss_outputs(self):
#
#         # First recording
#         # self.enable_records()
#
#         # Initialize system
#         self.initialize_system()
#
#         # Simulate to reach steady state
#         self.sim.RunMode = "Dynamic"
#         self.sim.options.TimeSettings.RecordHistory = False
#
#         self.sim.endtime = 40  # reasonable time to achieve steady state
#         self.sim.run(1)
#
#         print("Steady state reached!")
#
#         return np.array([self.col.Stage[24].y[1].Value,
#                          self.col.Stage[85].T.Value])
#
#     def step(self):
#
#         # Update inputs and step the simulation
#         self.col.Reflux.FmR.Value = self.current_input[0]
#         self.col.QRebR.Value = self.current_input[1]
#
#         # Run the system with the specified inputs
#         self.sim.Step(True)
#
#         # Change the current output
#         self.current_output = np.array([self.col.Stage[24].y[1].Value,
#                                         self.col.Stage[85].T.Value])
#
#     def close(self, snaps_path, prefix="snp"):
#
#         files = [os.path.join(snaps_path, file) for file in os.listdir(snaps_path)
#                  if os.path.isfile(os.path.join(snaps_path, file)) and file.startswith(prefix)]
#
#         files.sort(key=os.path.getctime)
#
#         if files:
#             for file in files:
#                 os.remove(file)
#                 print(f"Deleted the last created snapshot starting with '{prefix}': {file}")
#         else:
#             print(f"No files to delete that start with '{prefix}'.")
#
#         # self.ad.CloseDocument(False)
#         self.ad.Quit()

import win32com.client
import numpy as np
import os


class DistillationColumnAspen:
    def __init__(self, path, ss_inputs, initialization_point):

        # First initiate the Aspen Dynamic file simulation
        self.ad = win32com.client.DispatchEx("AD application")
        self.ad.NewDocument()
        self.ad.Visible = True
        self.ad.activate()
        self.ad.Maximize()

        # Open The path of simulation
        self.ad.openDocument(path)

        # Add all the directories of the simulation to python
        self.sim = self.ad.Simulation
        self.fsheet = self.sim.Flowsheet
        self.streams = self.fsheet.Streams
        self.block = self.fsheet.Blocks

        # Cache the main column block for faster access
        self.col = self.block("C2S")

        # Nominal condition
        self.initialization_point = initialization_point

        # Steady State inputs
        self.ss_inputs = ss_inputs

        # Define steady state outputs
        self.y_ss = self.ss_outputs()

        # Current inputs and outputs
        self.current_input = self.ss_inputs
        self.current_output = self.y_ss

        # Assigning Feed
        self.feed = self.streams("Feed")

    def enable_records(self):
        # Enabling records
        self.col.Reflux.FmR.Record = True
        self.col.QRebR.Record = True
        self.col.Stage(24).y("C2H6").Record = True
        self.col.Stage(85).T.Record = True
        print("Records have been enabled")

    def initialize_system(self):

        # Set initialization values
        feed = self.streams("Feed")
        feed.FmR.Value = self.initialization_point[0]
        feed.T.Value = self.initialization_point[1]
        feed.P.Value = self.initialization_point[2]
        feed.ZR("C2H4").Value = self.initialization_point[3]
        feed.ZR("C2H6").Value = self.initialization_point[4]

        hx = self.block("HX")
        hx.T.Spec = "Fixed"
        hx.T.Value = self.initialization_point[5]
        hx.QR.Spec = "Free"

        # Run Initialization
        # Control Values all to auto
        self.fsheet.TC.Cascade.Value = 0
        self.fsheet.TC.AutoMan.Value = 0
        self.fsheet.EAC.Cascade.Value = 0
        self.fsheet.EAC.AutoMan.Value = 0

        self.sim.RunMode = "Initialization"
        self.sim.Run(True)
        if self.sim.Successful:
            print("Initialization has been completed")
        else:
            print("Initialization has been failed")

        self.fsheet.TC.Cascade.Value = 1
        self.fsheet.TC.AutoMan.Value = 0
        self.fsheet.EAC.Cascade.Value = 1
        self.fsheet.EAC.AutoMan.Value = 0

        # Run Steady States (This is a step in Aspen dynamics. It doesn't mean steady state conditions)
        self.sim.RunMode = "Steady State"
        self.sim.Run(True)
        if self.sim.Successful:
            print("Steady State has been completed")
        else:
            print("Steady State has been failed")

        # Now make the system Open Loop
        for block_name in ["TC", "EAC"]:
            self.fsheet.RemoveBlock(block_name)
        for stream_name in ["S1", "S2", "S4", "S9"]:
            self.fsheet.RemoveStream(stream_name)

        print("System is in open loop condition now!")

        self.col.Reflux.FmR.Value = self.ss_inputs[0]
        self.col.QRebR.Value = self.ss_inputs[1]

    def ss_outputs(self):

        # First recording
        # self.enable_records()

        # Initialize system
        self.initialize_system()

        # Simulate to reach steady state
        self.sim.RunMode = "Dynamic"
        self.sim.options.TimeSettings.RecordHistory = False

        self.sim.endtime = 40  # reasonable time to achieve steady state
        self.sim.run(1)

        print("Steady state reached!")

        return np.array([self.col.Stage(24).y("C2H6").Value,
                         self.col.Stage(85).T.Value])

    def step(self, disturbances=None):

        # Update inputs and step the simulation
        self.col.Reflux.FmR.Value = self.current_input[0]
        self.col.QRebR.Value = self.current_input[1]

        # Run the system with the specified inputs
        self.sim.Step(True)

        # Change the current output
        self.current_output = np.array([self.col.Stage(24).y("C2H6").Value,
                                        self.col.Stage(85).T.Value])

        # Disturbance should be applied after stepping the system
        if disturbances is not None:
            self.feed.FmR.Value = disturbances[0]

    def close(self, snaps_path, prefix="snp"):

        files = [os.path.join(snaps_path, file) for file in os.listdir(snaps_path)
                 if os.path.isfile(os.path.join(snaps_path, file)) and file.startswith(prefix)]

        files.sort(key=os.path.getctime)

        if files:
            for file in files:
                os.remove(file)
                print(f"Deleted the last created snapshot starting with '{prefix}': {file}")
        else:
            print(f"No files to delete that start with '{prefix}'.")

        self.ad.CloseDocument(False)
        self.ad.Quit()
