#!/usr/bin/env python3
"""
Very simple script to test Arduino serial communication
This script just opens the serial port and prints any data received
"""
import sys
import time
import serial
import serial.tools.list_ports

def list_ports():
    """List all available serial ports"""
    ports = list(serial.tools.list_ports.comports())
    print(f"Found {len(ports)} serial ports:")
    for i, port in enumerate(ports):
        print(f"  {i+1}. {port.device} - {port.description}")
    return ports

def test_port(port_name, baud_rate=115200, timeout_secs=30):
    """Test serial communication with the specified port"""
    print(f"\nTesting communication with {port_name} at {baud_rate} baud...")
    
    try:
        # Open serial port
        ser = serial.Serial(port_name, baud_rate, timeout=1)
        print("Port opened successfully!")
        
        # Flush any pending data
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        # Send a test message
        test_message = "PYTHON_TEST\n"
        print(f"Sending test message: {test_message.strip()}")
        ser.write(test_message.encode('utf-8'))
        
        # Read responses
        print("\nReading serial data:")
        print("-" * 40)
        
        # Read for specified timeout or until Ctrl+C
        start_time = time.time()
        try:
            while (time.time() - start_time) < timeout_secs:
                if ser.in_waiting:
                    # Try to read a line (until newline character)
                    line = ser.readline()
                    
                    # Try different decodings if utf-8 fails
                    try:
                        decoded = line.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        try:
                            decoded = line.decode('latin-1').strip()
                        except:
                            decoded = f"[Binary data: {' '.join([f'{b:02x}' for b in line])}]"
                    
                    # Print the received data with timestamp
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] {decoded}")
                    
                    # Also print raw bytes for debugging
                    if len(line) > 0:
                        print(f"  Raw bytes: {' '.join([f'{b:02x}' for b in line])}")
                
                # Small delay to prevent CPU overuse
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nTest terminated by user")
        
    except serial.SerialException as e:
        print(f"Error opening/using serial port: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("\nSerial port closed")

def main():
    """Main function"""
    print("Arduino Serial Communication Test")
    print("=" * 40)
    
    # List available ports
    ports = list_ports()
    
    if not ports:
        print("No serial ports found. Please check connections.")
        return
    
    # Let user select a port
    port_name = None
    if len(ports) == 1:
        port_name = ports[0].device
        print(f"\nUsing the only available port: {port_name}")
    else:
        while True:
            try:
                choice = input("\nEnter port number to test, or 'q' to quit: ")
                if choice.lower() == 'q':
                    return
                
                port_index = int(choice) - 1
                if 0 <= port_index < len(ports):
                    port_name = ports[port_index].device
                    break
                else:
                    print("Invalid port number.")
            except ValueError:
                print("Please enter a valid number.")
    
    # Test the selected port
    test_port(port_name)

if __name__ == "__main__":
    main()