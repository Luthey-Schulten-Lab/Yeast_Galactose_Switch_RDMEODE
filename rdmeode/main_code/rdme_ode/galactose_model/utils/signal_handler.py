"""Signal handling for graceful shutdown"""

import signal
import sys


def setup_signal_handler(solver_container):
    """Setup signal handler for graceful simulation shutdown

    Args:
        solver_container (dict): Dictionary containing 'solver' key with solver instance
    """

    def signal_handler(signum, frame):
        """Handle interrupt signals"""
        print("\n" + "=" * 80)
        print("Interrupt received, stopping simulation gracefully...")
        print("=" * 80)

        solver = solver_container.get('solver')
        if solver is not None:
            # Close file handles
            if hasattr(solver, 'save_cts_by_region_handle'):
                solver.save_cts_by_region_handle.close()
                print(f"Closed: {solver.save_cts_by_region_file}")

            if hasattr(solver, 'save_ode_data_handle'):
                solver.save_ode_data_handle.close()
                print(f"Closed: {solver.save_ode_data_file}")

        print("Exiting...")
        sys.exit(0)

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
