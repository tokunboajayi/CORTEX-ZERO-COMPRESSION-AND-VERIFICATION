# NS-ARC Synthesis Script for Xilinx Vivado
# Usage: vivado -mode batch -source synthesize.tcl

# 1. Project Setup
set project_name "ns_arc_fpga"
set part "xcu250-figd2104-2L-e" ;# Alveo U250

create_project -force $project_name ./build -part $part

# 2. Add Sources
add_files -norecurse ./ns_arc_fpga.v
update_compile_order -fileset sources_1

# 3. Define Constraints (Clock 300MHz)
# Create a dummy constraint file in memory
set constraints_file [open "constraints.xdc" w]
puts $constraints_file "create_clock -period 3.333 -name sys_clk [get_ports clk]"
close $constraints_file
add_files -fileset constrs_1 ./constraints.xdc

# 4. Synthesis
puts "--- Starting Synthesis ---"
synth_design -top ns_arc_accelerator -part $part
write_checkpoint -force $project_name_synth.dcp
report_utilization -file utilization_synth.rpt

# 5. Implementation (Place & Route)
puts "--- Starting Implementation ---"
opt_design
place_design
route_design
write_checkpoint -force $project_name_impl.dcp
report_timing_summary -file timing.rpt

# 6. Bitstream Generation
puts "--- Generating Bitstream ---"
write_bitstream -force ns_arc_demon.bit

puts "--- DONE. Bitstream ready at ns_arc_demon.bit ---"
