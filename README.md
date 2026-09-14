# Operating Systems – Threading Projects

This repository contains two projects developed to demonstrate important concepts of **Operating Systems**, particularly multithreading, synchronization, shared resources, task scheduling, and concurrent execution.

The projects include:

1. **Producer–Consumer Problem using Java Threads and Semaphores**
2. **100 × 100 Matrix Multiplication using Threads and TensorFlow**

---

# Task 1 – Producer Consumer Problem

## Objective

To implement the **Producer–Consumer synchronization problem** using multiple threads in Java.

The implementation demonstrates how producers and consumers safely access a shared bounded buffer using synchronization mechanisms.

---

## Features

- Two Producer threads
- Two Consumer threads
- Shared bounded buffer of size 5
- Semaphore-based synchronization
- Mutual exclusion while accessing the buffer
- Detection of full and empty buffer conditions
- Producer waiting when the buffer is full
- Consumer waiting when the buffer is empty
- Random item generation
- Execution time calculation
- Throughput calculation

---

## Synchronization Mechanism

Three semaphores are used:

| Semaphore | Purpose |
|---|---|
| `empty` | Keeps track of empty positions in the buffer |
| `full` | Keeps track of filled positions in the buffer |
| `mutex` | Ensures mutual exclusion while accessing the shared buffer |

The initial values are:

```text
empty = 5
full  = 0
mutex = 1
```

---

## Working

### Producer

A producer performs the following steps:

```text
Wait for an empty buffer slot
        ↓
Acquire mutex
        ↓
Produce and add an item
        ↓
Release mutex
        ↓
Signal that a filled slot is available
```

### Consumer

A consumer performs:

```text
Wait for an available item
        ↓
Acquire mutex
        ↓
Remove an item from the buffer
        ↓
Release mutex
        ↓
Signal that an empty slot is available
```

---

## Example Output

```text
Producer-1 produced: 45
Buffer: [45]

Producer-2 produced: 72
Buffer: [45, 72]

Consumer-1 consumed: 45
Buffer: [72]

Producer-1 produced: 36
Buffer: [72, 36]

Consumer-2 consumed: 72
Buffer: [36]
```

When the buffer becomes full:

```text
Producer-1 is WAITING because buffer is FULL...
```

When the buffer becomes empty:

```text
Consumer-2 is WAITING because buffer is EMPTY...
```

At the end of execution, the program displays:

```text
--------------------------------
        EXECUTION SUMMARY
--------------------------------
Total items produced : 10
Total items consumed : 10
Execution time       : 5.12 seconds
Throughput           : 1.95 items/second
--------------------------------
```

---

## Technologies Used

- Java
- Java Threads
- `Semaphore`
- `AtomicInteger`
- Queue / LinkedList

---

## How to Run Task 1

Compile the Java program:

```bash
javac ProducerConsumer.java
```

Run the program:

```bash
java ProducerConsumer
```

---

# Task 2 – Multithreaded Matrix Multiplication using TensorFlow

## Objective

To perform multiplication of two **100 × 100 matrices** using multiple threads and TensorFlow.

The result matrix also contains:

```text
100 × 100 = 10,000 result cells
```

Each result cell requires a row-column dot product.

Therefore, the complete matrix multiplication performs:

```text
100 × 100 × 100
= 1,000,000 scalar multiplications
```

---

## Features

- Two randomly generated 100 × 100 matrices
- 10,000 matrix-cell calculation tasks
- 8 manually created worker threads
- Shared task queue
- Shared result queue
- TensorFlow-based multiplication
- Thread-safe workload statistics
- Execution-time measurement
- Cell throughput measurement
- Scalar multiplication-rate measurement
- Work-distribution analysis for all threads
- Result verification using `tf.matmul`
- Animated GIF output
- Final PNG visualization

---

## Threading Architecture

Instead of creating thousands of physical threads, the program creates **8 worker threads**.

The 10,000 result-cell tasks are stored in a shared task queue.

```text
                  TASK QUEUE
         C[0][0], C[0][1], C[0][2] ...
                       |
        ---------------------------------
        |       |       |       |       |
        v       v       v       v       v
     Worker   Worker  Worker   ...    Worker
       1        2       3               8
        |       |       |               |
        -------- TensorFlow -------------
                       |
                       v
                  RESULT QUEUE
                       |
                       v
                 Result Matrix C
```

Each worker repeatedly takes a matrix-cell task from the queue and performs the corresponding row-column multiplication. The implementation uses a shared task queue and manually created worker threads rather than relying only on a thread-pool abstraction. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

---

## Matrix Cell Calculation

For every element:

```text
C[i][j]
```

one row from Matrix A is multiplied with one column from Matrix B.

For example:

```text
C[2][4]

= A[2][0] × B[0][4]
+ A[2][1] × B[1][4]
+ ...
+ A[2][99] × B[99][4]
```

TensorFlow performs the element-wise multiplication and reduction for each cell using `tf.multiply()` and `tf.reduce_sum()`. :contentReference[oaicite:3]{index=3}

---

## Result Verification

After threaded multiplication is complete, the result is independently calculated using:

```python
tf.matmul(tensor_a, tensor_b)
```

The two matrices are compared using:

```python
np.allclose()
```

If both results match:

```text
Verification : PASSED
```

This provides an additional correctness check for the threaded implementation. :contentReference[oaicite:4]{index=4}

---

# Animated Output

The animation shows:

- Matrix A
- Matrix B
- Result Matrix C
- Current row selected from Matrix A
- Current column selected from Matrix B
- Corresponding result cell in Matrix C
- Work completed by each of the 8 worker threads
- Overall progress of matrix multiplication

<p align="center">
  <img src="matrix_thread_animation.gif" alt="Threaded Matrix Multiplication Animation" width="850">
</p>

The GIF visualizes the recorded order in which threaded matrix-cell tasks were completed.

---

## Final Output

<p align="center">
  <img src="matrix_thread_summary.png" alt="Matrix Multiplication Final Output" width="850">
</p>

The visualization contains the three matrices along with the final thread-work distribution. The Python implementation saves both the GIF and PNG automatically. :contentReference[oaicite:5]{index=5}

---

## Execution Summary

The program also displays performance information such as:

```text
=================================================================
                    EXECUTION SUMMARY
=================================================================
Total Result Cells      : 10,000
Scalar Multiplications  : 1,000,000
Worker Threads          : 8
Execution Time          : ... seconds
Cell Throughput         : ... cells/sec
Multiplication Rate     : ... operations/sec
TensorFlow Check Time   : ... seconds
Verification            : PASSED
=================================================================
```

It additionally displays the number of result cells processed by every worker thread. :contentReference[oaicite:6]{index=6}

---

## Technologies Used

- Python
- TensorFlow
- NumPy
- Python `threading`
- `queue.Queue`
- Matplotlib
- Pillow
- GIF animation

---

## Requirements

Install the required Python libraries:

```bash
pip install tensorflow numpy matplotlib pillow
```

Python 3.10 is recommended for this project.

---

## How to Run Task 2

Run the Python program:

```bash
python matrix_multiplication.py
```

After successful execution, the following files are generated:

```text
matrix_thread_animation.gif
matrix_thread_summary.png
```

---

# Project Structure

```text
Operating-System-Projects/
│
├── ProducerConsumer.java
│
├── matrix_multiplication.py
│
├── matrix_thread_animation.gif
│
├── matrix_thread_summary.png
│
└── README.md
```

---

# Concepts Demonstrated

The two projects demonstrate several Operating Systems concepts:

- Multithreading
- Thread synchronization
- Semaphores
- Mutual exclusion
- Critical sections
- Shared resources
- Producer–Consumer synchronization
- Worker-thread model
- Shared task queues
- Concurrent task execution
- Thread workload distribution
- Performance measurement

---

# Conclusion

The first task demonstrates synchronization between multiple producers and consumers using semaphores and a bounded shared buffer.

The second task extends multithreading to a computational problem by distributing 100 × 100 matrix multiplication tasks among multiple worker threads. TensorFlow is used for numerical computation, while performance statistics and animation provide a clear representation of how the threaded computation is carried out.
