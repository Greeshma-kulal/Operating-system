import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;

public class ProducerConsumer {

    static final int BUFFER_SIZE = 5;
    static final int ITEMS_PER_THREAD = 5;

    static Queue<Integer> buffer = new LinkedList<>();

    static Semaphore empty = new Semaphore(BUFFER_SIZE);
    static Semaphore full = new Semaphore(0);
    static Semaphore mutex = new Semaphore(1);

    static Random random = new Random();

    static AtomicInteger totalProduced = new AtomicInteger(0);
    static AtomicInteger totalConsumed = new AtomicInteger(0);

    public static void main(String[] args) {

        long startTime = System.currentTimeMillis();

        Thread producer1 = createProducer("Producer-1");
        Thread producer2 = createProducer("Producer-2");

        Thread consumer1 = createConsumer("Consumer-1");
        Thread consumer2 = createConsumer("Consumer-2");

        producer1.start();
        producer2.start();

        consumer1.start();
        consumer2.start();

        try {
            producer1.join();
            producer2.join();

            consumer1.join();
            consumer2.join();

        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        long endTime = System.currentTimeMillis();

        double executionTime = (endTime - startTime) / 1000.0;

        double throughput = totalConsumed.get() / executionTime;

        System.out.println("\n--------------------------------");
        System.out.println("        EXECUTION SUMMARY");
        System.out.println("--------------------------------");

        System.out.println("Total items produced : " + totalProduced.get());
        System.out.println("Total items consumed : " + totalConsumed.get());

        System.out.printf(
                "Execution time       : %.2f seconds%n",
                executionTime);

        System.out.printf(
                "Throughput           : %.2f items/second%n",
                throughput);

        System.out.println("--------------------------------");
    }


    static Thread createProducer(String name) {

        return new Thread(() -> {

            for (int i = 1; i <= ITEMS_PER_THREAD; i++) {

                try {

                    if (empty.availablePermits() == 0) {

                        System.out.println(
                                "\n" + name +
                                " is WAITING because buffer is FULL...");
                    }

                    empty.acquire();

                    mutex.acquire();

                    int item = random.nextInt(100) + 1;

                    buffer.add(item);

                    totalProduced.incrementAndGet();

                    System.out.println(
                            "\n" + name +
                            " produced: " + item);

                    System.out.println(
                            "Buffer: " + buffer);

                    mutex.release();

                    full.release();

                    Thread.sleep(500);

                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            System.out.println(
                    name + " finished producing.");
        });
    }


    static Thread createConsumer(String name) {

        return new Thread(() -> {

            for (int i = 1; i <= ITEMS_PER_THREAD; i++) {

                try {

                    if (full.availablePermits() == 0) {

                        System.out.println(
                                "\n" + name +
                                " is WAITING because buffer is EMPTY...");
                    }

                    full.acquire();

                    mutex.acquire();

                    int item = buffer.remove();

                    totalConsumed.incrementAndGet();

                    System.out.println(
                            "\n" + name +
                            " consumed: " + item);

                    System.out.println(
                            "Buffer: " + buffer);

                    mutex.release();

                    empty.release();

                    Thread.sleep(1000);

                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            System.out.println(
                    name + " finished consuming.");
        });
    }
}
