#!/bin/bash
# filepath: /Users/DAADAMS/Other/query-data-predictor/run_with_memory_monitor.sh

# Configuration
PYTHON_CMD="pytest -vvs tests/test_dataset_creator.py::TestDatasetCreator::test_full_build"  # Replace with your actual command
INTERVAL=5                # Check interval in seconds
MAX_MEMORY_PERCENT=9m     # Terminate if memory exceeds this percentage (set to 0 to disable)
LOG_FILE="memory_usage.log"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DETAIL_LOG="memory_detail_${TIMESTAMP}.log"

# Clear previous log
echo "Time,PID,Memory(MB),CPU(%)" > "$LOG_FILE"
echo "Memory monitoring started at $(date)" > "$DETAIL_LOG"
echo "Process command: $PYTHON_CMD" >> "$DETAIL_LOG"
echo "-----------------------------------------" >> "$DETAIL_LOG"

# Start the Python process
echo "Starting Python process: $PYTHON_CMD"
$PYTHON_CMD &
PID=$!
echo "Process started with PID: $PID"

# Function to get memory usage in MB
get_memory_usage() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        ps -o rss= -p $1 | awk '{print $1/1024}'
    else
        # Linux
        ps -o rss= -p $1 | awk '{print $1/1024}'
    fi
}

# Function to get CPU usage
get_cpu_usage() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        ps -o %cpu= -p $1
    else
        # Linux
        ps -o %cpu= -p $1
    fi
}

# Function to check system memory usage (percent)
get_system_memory_percent() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - this is an approximation
        vm_stat | grep "Pages free:" | awk '{print 100-$3*4096/4/1024/1024/16*100}'
    else
        # Linux
        free | grep Mem | awk '{print $3/$2 * 100.0}'
    fi
}

# Monitor the process
echo "Monitoring memory usage every $INTERVAL seconds"
echo "Press Ctrl+C to stop monitoring"

while kill -0 $PID 2>/dev/null; do
    MEM=$(get_memory_usage $PID)
    CPU=$(get_cpu_usage $PID)
    SYS_MEM=$(get_system_memory_percent)
    TIME=$(date +"%H:%M:%S")
    
    # Log data
    echo "$TIME,$PID,$MEM,$CPU" >> "$LOG_FILE"
    
    # Detailed memory info
    echo "Time: $TIME - Memory: ${MEM}MB - CPU: ${CPU}% - System Memory: ${SYS_MEM}%" >> "$DETAIL_LOG"
    
    # Check if memory limit is exceeded
    if [ $MAX_MEMORY_PERCENT -gt 0 ] && [ $(echo "$SYS_MEM > $MAX_MEMORY_PERCENT" | bc) -eq 1 ]; then
        echo "WARNING: Memory usage exceeded threshold ($SYS_MEM% > $MAX_MEMORY_PERCENT%)" >> "$DETAIL_LOG"
        echo "Terminating process $PID" >> "$DETAIL_LOG"
        echo "Memory limit exceeded! Terminating process."
        kill -15 $PID
        sleep 2
        if kill -0 $PID 2>/dev/null; then
            echo "Process didn't terminate gracefully, forcing kill" >> "$DETAIL_LOG"
            kill -9 $PID
        fi
        break
    fi
    
    # Get more detailed memory info on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "----- Process Details -----" >> "$DETAIL_LOG"
        ps -xm -o pid,ppid,user,%cpu,%mem,vsz,rss,tt,state,start,time,command -p $PID >> "$DETAIL_LOG"
        echo "------------------------" >> "$DETAIL_LOG"
    fi
    
    sleep $INTERVAL
done

if ! kill -0 $PID 2>/dev/null; then
    echo "Process $PID has completed."
    echo "Process completed at $(date)" >> "$DETAIL_LOG"
else
    echo "Process $PID is still running. Monitoring stopped."
    echo "Monitoring stopped at $(date)" >> "$DETAIL_LOG"
fi

echo "Memory usage log saved to $LOG_FILE"
echo "Detailed log saved to $DETAIL_LOG"

# Optional: Generate a simple plot if gnuplot is available
if command -v gnuplot > /dev/null; then
    PLOT_FILE="memory_plot_${TIMESTAMP}.png"
    echo "Generating memory usage plot..."
    gnuplot <<EOF
    set terminal png size 800,600
    set output "$PLOT_FILE"
    set title "Memory Usage Over Time"
    set xlabel "Time"
    set ylabel "Memory (MB)"
    set y2label "CPU (%)"
    set y2tics
    set grid
    set datafile separator ","
    plot "$LOG_FILE" using 3 with lines title "Memory (MB)" axis x1y1, \
         "$LOG_FILE" using 4 with lines title "CPU (%)" axis x1y2
EOF
    echo "Plot saved to $PLOT_FILE"
fi