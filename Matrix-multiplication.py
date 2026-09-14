import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import queue
import threading

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle




N = 100
NUMBER_OF_THREADS = 8

TOTAL_CELLS = N * N
TOTAL_MULTIPLICATIONS = N * N * N

CELLS_PER_FRAME = 250

GIF_FILE = "matrix_thread_animation.gif"
IMAGE_FILE = "matrix_thread_summary.png"



try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass



random_generator = np.random.default_rng(25)

matrix_a = random_generator.integers(
    1,
    10,
    size=(N, N)
).astype(np.float32)

matrix_b = random_generator.integers(
    1,
    10,
    size=(N, N)
).astype(np.float32)


tensor_a = tf.constant(
    matrix_a,
    dtype=tf.float32
)

tensor_b = tf.constant(
    matrix_b,
    dtype=tf.float32
)




result_matrix = np.zeros(
    (N, N),
    dtype=np.float32
)



task_queue = queue.Queue()

result_queue = queue.Queue()



thread_counts = [0] * NUMBER_OF_THREADS

stats_lock = threading.Lock()



completion_history = []



def calculate_cell(row, column):


    row_values = tensor_a[row, :]


    column_values = tensor_b[:, column]


    products = tf.multiply(
        row_values,
        column_values
    )

    # Sum the multiplied values
    answer = tf.reduce_sum(
        products
    )

    return float(answer.numpy())




def worker(worker_id):

    while True:

        task = task_queue.get()

        
        if task is None:
            task_queue.task_done()
            break

        row, column = task

        value = calculate_cell(
            row,
            column
        )

        
        with stats_lock:
            thread_counts[worker_id] += 1

        
        result_queue.put(
            (
                row,
                column,
                value,
                worker_id
            )
        )

        task_queue.task_done()




def run_threaded_multiplication():

    print("=" * 65)
    print("   MULTITHREADED MATRIX MULTIPLICATION USING TENSORFLOW")
    print("=" * 65)

    print(f"Matrix A               : {N} x {N}")
    print(f"Matrix B               : {N} x {N}")
    print(f"Result Matrix          : {N} x {N}")

    print(
        f"Worker Threads         : "
        f"{NUMBER_OF_THREADS}"
    )

    print(
        f"Threaded Cell Tasks    : "
        f"{TOTAL_CELLS:,}"
    )

    print(
        f"Scalar Multiplications : "
        f"{TOTAL_MULTIPLICATIONS:,}"
    )

    print("=" * 65)



    for row in range(N):

        for column in range(N):

            task_queue.put(
                (row, column)
            )

    # One stop signal for every worker
    for _ in range(NUMBER_OF_THREADS):

        task_queue.put(None)



    threads = []

    start_time = time.perf_counter()

    for worker_id in range(NUMBER_OF_THREADS):

        thread = threading.Thread(
            target=worker,
            args=(worker_id,),
            name=f"Worker-{worker_id + 1}"
        )

        threads.append(thread)

        thread.start()



    completed = 0

    while completed < TOTAL_CELLS:

        row, column, value, worker_id = (
            result_queue.get()
        )

        result_matrix[
            row,
            column
        ] = value

        completion_history.append(
            (
                row,
                column,
                value,
                worker_id
            )
        )

        completed += 1

        # Terminal progress
        if completed % 1000 == 0:

            percentage = (
                completed
                / TOTAL_CELLS
            ) * 100

            print(
                f"Completed "
                f"{completed:,}/{TOTAL_CELLS:,} "
                f"cells ({percentage:.0f}%)"
            )

    # --------------------------------------------------------
    # WAIT FOR ALL THREADS
    # --------------------------------------------------------

    for thread in threads:
        thread.join()

    end_time = time.perf_counter()

    execution_time = (
        end_time - start_time
    )

    return execution_time




def verify_result():

    start = time.perf_counter()

    correct_result = tf.matmul(
        tensor_a,
        tensor_b
    ).numpy()

    verification_time = (
        time.perf_counter()
        - start
    )

    verified = np.allclose(
        result_matrix,
        correct_result
    )

    return verified, verification_time




def print_summary(
    execution_time,
    verified,
    verification_time
):

    cell_throughput = (
        TOTAL_CELLS
        / execution_time
    )

    multiplication_rate = (
        TOTAL_MULTIPLICATIONS
        / execution_time
    )

    print("\n")
    print("=" * 65)
    print("                    EXECUTION SUMMARY")
    print("=" * 65)

    print(
        f"Total Result Cells      : "
        f"{TOTAL_CELLS:,}"
    )

    print(
        f"Scalar Multiplications  : "
        f"{TOTAL_MULTIPLICATIONS:,}"
    )

    print(
        f"Worker Threads          : "
        f"{NUMBER_OF_THREADS}"
    )

    print(
        f"Execution Time          : "
        f"{execution_time:.4f} seconds"
    )

    print(
        f"Cell Throughput         : "
        f"{cell_throughput:,.2f} cells/sec"
    )

    print(
        f"Multiplication Rate     : "
        f"{multiplication_rate:,.2f} operations/sec"
    )

    print(
        f"TensorFlow Check Time   : "
        f"{verification_time:.6f} seconds"
    )

    print(
        "Verification            :",
        "PASSED"
        if verified
        else "FAILED"
    )

    print("\nWORK DONE BY EACH THREAD")
    print("-" * 35)

    for i, count in enumerate(
        thread_counts
    ):

        print(
            f"Worker-{i + 1} : "
            f"{count} cells"
        )

    print("\nResult Matrix C - First 5 x 5:")

    print(
        result_matrix[
            :5,
            :5
        ].astype(int)
    )

    print("=" * 65)



def create_visual_outputs():

    print("\nCreating animation...")

    # Initially C is hidden
    visible_result = np.full(
        (N, N),
        np.nan,
        dtype=np.float32
    )



    fig = plt.figure(
        figsize=(11, 6)
    )

    layout = fig.add_gridspec(
        2,
        3,
        height_ratios=[3.2, 1],
        hspace=0.45,
        wspace=0.32
    )



    ax_a = fig.add_subplot(
        layout[0, 0]
    )

    ax_a.imshow(
        matrix_a,
        aspect="equal"
    )

    ax_a.set_title(
        "Matrix A",
        fontsize=10,
        fontweight="bold"
    )

    ax_a.set_xlabel(
        "Columns",
        fontsize=8
    )

    ax_a.set_ylabel(
        "Rows",
        fontsize=8
    )

    ax_a.tick_params(
        labelsize=7
    )



    ax_b = fig.add_subplot(
        layout[0, 1]
    )

    ax_b.imshow(
        matrix_b,
        aspect="equal"
    )

    ax_b.set_title(
        "Matrix B",
        fontsize=10,
        fontweight="bold"
    )

    ax_b.set_xlabel(
        "Columns",
        fontsize=8
    )

    ax_b.set_ylabel(
        "Rows",
        fontsize=8
    )

    ax_b.tick_params(
        labelsize=7
    )


    ax_c = fig.add_subplot(
        layout[0, 2]
    )

    result_image = ax_c.imshow(
        np.ma.masked_invalid(
            visible_result
        ),
        aspect="equal",
        vmin=np.min(result_matrix),
        vmax=np.max(result_matrix)
    )

    ax_c.set_title(
        "Result Matrix C",
        fontsize=10,
        fontweight="bold"
    )

    ax_c.set_xlabel(
        "Columns",
        fontsize=8
    )

    ax_c.set_ylabel(
        "Rows",
        fontsize=8
    )

    ax_c.tick_params(
        labelsize=7
    )


    ax_threads = fig.add_subplot(
        layout[1, :]
    )

    worker_names = [
        f"T{i + 1}"
        for i in range(NUMBER_OF_THREADS)
    ]

    bars = ax_threads.bar(
        worker_names,
        [0] * NUMBER_OF_THREADS
    )

    ax_threads.set_title(
        "Thread Work Distribution",
        fontsize=9,
        fontweight="bold"
    )

    ax_threads.set_xlabel(
        "Worker Threads",
        fontsize=8
    )

    ax_threads.set_ylabel(
        "Completed Cells",
        fontsize=8
    )

    ax_threads.tick_params(
        labelsize=7
    )

    max_work = max(thread_counts)

    ax_threads.set_ylim(
        0,
        max_work * 1.15
    )


    row_box = Rectangle(
        (-0.5, -0.5),
        N,
        1,
        fill=False,
        edgecolor="red",
        linewidth=1.8
    )

    column_box = Rectangle(
        (-0.5, -0.5),
        1,
        N,
        fill=False,
        edgecolor="red",
        linewidth=1.8
    )

    result_box = Rectangle(
        (-0.5, -0.5),
        1,
        1,
        fill=False,
        edgecolor="red",
        linewidth=2
    )

    ax_a.add_patch(
        row_box
    )

    ax_b.add_patch(
        column_box
    )

    ax_c.add_patch(
        result_box
    )


    fig.suptitle(
        "100 x 100 Multithreaded TensorFlow Matrix Multiplication",
        fontsize=13,
        fontweight="bold"
    )

    status_text = fig.text(
        0.5,
        0.01,
        "Starting animation...",
        ha="center",
        fontsize=8
    )


    frame_count = int(
        np.ceil(
            TOTAL_CELLS
            / CELLS_PER_FRAME
        )
    )


    def update(frame):

        end = min(
            (frame + 1) * CELLS_PER_FRAME,
            TOTAL_CELLS
        )



        visible_result[:] = np.nan

        frame_thread_counts = (
            [0] * NUMBER_OF_THREADS
        )

        current_row = 0
        current_column = 0
        current_worker = 0

        for index in range(end):

            (
                row,
                column,
                value,
                worker_id
            ) = completion_history[index]

            visible_result[
                row,
                column
            ] = value

            frame_thread_counts[
                worker_id
            ] += 1

            current_row = row
            current_column = column
            current_worker = worker_id

        # ----------------------------------------------------
        # RESULT MATRIX
        # ----------------------------------------------------

        result_image.set_data(
            np.ma.masked_invalid(
                visible_result
            )
        )

     

        row_box.set_y(
            current_row - 0.5
        )

        column_box.set_x(
            current_column - 0.5
        )

        result_box.set_xy(
            (
                current_column - 0.5,
                current_row - 0.5
            )
        )

       

        for bar, value in zip(
            bars,
            frame_thread_counts
        ):

            bar.set_height(
                value
            )

     

        progress = (
            end
            / TOTAL_CELLS
        ) * 100

        status_text.set_text(
            f"Progress: {progress:.0f}%   |   "
            f"Completed: {end:,}/{TOTAL_CELLS:,}   |   "
            f"Cell: C[{current_row}][{current_column}]   |   "
            f"Worker-{current_worker + 1}"
        )

        return (
            result_image,
            row_box,
            column_box,
            result_box,
            status_text,
            *bars
        )

  

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=120,
        repeat=False,
        cache_frame_data=False
    )

   

    print(
        f"Saving GIF as: "
        f"{GIF_FILE}"
    )

    gif_writer = PillowWriter(
        fps=8
    )

    animation.save(
        GIF_FILE,
        writer=gif_writer,
        dpi=90
    )

    print("GIF created successfully.")

   
    update(
        frame_count - 1
    )

    fig.savefig(
        IMAGE_FILE,
        dpi=160,
        bbox_inches="tight"
    )

    print(
        f"Image saved as: "
        f"{IMAGE_FILE}"
    )

    
    plt.show()

    plt.close(fig)



def main():

    execution_time = (
        run_threaded_multiplication()
    )

    verified, verification_time = (
        verify_result()
    )

    print_summary(
        execution_time,
        verified,
        verification_time
    )

    create_visual_outputs()

    print("\nGenerated files:")
    print("1.", GIF_FILE)
    print("2.", IMAGE_FILE)

    print("\nProgram completed successfully.")




if __name__ == "__main__":
    main()
