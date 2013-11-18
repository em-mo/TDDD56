/*
 * stack_test.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 *
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>

#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
    printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
    test_setup();\
    if(test())\
    {\
        printf("passed\n");\
    }\
    else\
    {\
        printf("failed\n");\
    }\
    test_teardown();

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

stack_t *stack;
data_t data;

stack_t **prealloc;


void
test_init()
{
    prealloc = (stack_t **) malloc(sizeof(stack_t *) * NB_THREADS);

    int i;
    for (i = 0; i < NB_THREADS; ++i)
    {
        prealloc[i] = (stack_t *) malloc(sizeof(stack_t) * MAX_PUSH_POP);
    }
}

void
test_setup()
{
    // Allocate and initialize your test stack before each test
    data = DATA_VALUE;
    stack = stack_alloc();
    stack_init(stack, DATA_SIZE);
}

void
test_teardown()
{
    // Do not forget to free your stacks after each test
    // to avoid memory leaks as now
    while (stack != NULL)
    {
        stack_t *tmp = stack;
        stack = stack->next;
        free(tmp);
    }
}

void
test_finalize()
{
    // Destroy properly your test batch
}
typedef struct thread_test_push_args thread_test_push_args_t;

struct thread_test_push_args
{
    int id;
};

void *
thread_test_stack_push(void *arg)
{
    thread_test_push_args_t *args = (thread_test_push_args_t *) arg;
    int i;
    for (i = 0; i < MAX_PUSH_POP; i++)
    {
        stack_push(&stack, &prealloc[args->id][i]);
    }
    return NULL;
}

void *
thread_test_stack_pop(void *data)
{
    int i;
    for (i = 0; i < MAX_PUSH_POP; i++)
    {
        stack_pop(&stack, data);
    }
    return NULL;
}

int
test_push_safe()
{
    // Make sure your stack remains in a good state with expected content when
    // several threads push concurrently to it

    pthread_attr_t attr;
    pthread_t thread[NB_THREADS];
    pthread_mutexattr_t mutex_attr;
    size_t counter;
    int i, success;

    counter = 0;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_mutexattr_init(&mutex_attr);

    thread_test_push_args_t args[NB_THREADS];
    for (i = 0; i < NB_THREADS; i++)
    {
        args[i].id  = i;
        pthread_create(&thread[i], &attr, &thread_test_stack_push, &args[i]);
    }


    for (i = 0; i < NB_THREADS; i++)
    {
        pthread_join(thread[i], NULL);
    }

    while (stack->next != NULL)
    {
        counter++;
        stack = stack->next;
    }

    success = counter == (size_t)(NB_THREADS * MAX_PUSH_POP);

    if (!success)
    {
        printf("Got %ti, expected %i\n", counter, NB_THREADS * MAX_PUSH_POP);
    }

    assert(success);

    return success;
}

int
test_pop_safe()
{
    // Same as the test above for parallel pop operation
    pthread_attr_t attr;
    pthread_t thread[NB_THREADS];
    pthread_mutexattr_t mutex_attr;
    size_t counter;
    int i, success;

    counter = 0;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_mutexattr_init(&mutex_attr);


    thread_test_push_args_t arg[NB_THREADS];
    for (i = 0; i < NB_THREADS; i++)
    {
        arg[i].id  = i;
        pthread_create(&thread[i], &attr, &thread_test_stack_push, &arg[i]);
    }

    for (i = 0; i < NB_THREADS; i++)
    {
        pthread_join(thread[i], NULL);
    }

    for (i = 0; i < NB_THREADS; i++)
    {
        int x;
        pthread_create(&thread[i], &attr, &thread_test_stack_pop, &x);
    }

    for (i = 0; i < NB_THREADS; i++)
    {
        pthread_join(thread[i], NULL);
    }

    while (stack->next != NULL)
    {
        counter++;
        stack = stack->next;
    }

    success = counter == 0;

    if (!success)
    {
        printf("Got %ti, expected %i\n", counter, NB_THREADS * MAX_PUSH_POP);
    }

    assert(success);

    return success;
}


void *
thread_measure_stack_push(void *arg)
{
    thread_test_push_args_t *args = (thread_test_push_args_t *) arg;
    int i;
    for (i = 0; i < MAX_PUSH_POP/NB_THREADS; i++)
    {
        stack_push(&stack, &prealloc[args->id][i]);
    }
    return NULL;
}

void *
thread_measure_stack_pop(void *data)
{
    int i;
    for (i = 0; i < MAX_PUSH_POP/NB_THREADS; i++)
    {
        stack_pop(&stack, data);
    }
    return NULL;
}

void
measure_pop()
{
    // Same as the test above for parallel pop operation
    pthread_attr_t attr;
    pthread_t thread[NB_THREADS];
    pthread_mutexattr_t mutex_attr;
    int i;

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_mutexattr_init(&mutex_attr);

    for (i = 0; i < NB_THREADS; i++)
    {
        int x;
        pthread_create(&thread[i], &attr, &thread_measure_stack_pop, &x);
    }

    for (i = 0; i < NB_THREADS; i++)
    {
        pthread_join(thread[i], NULL);
    }
    return;
}


void
measure_push()
{
    // Make sure your stack remains in a good state with expected content when
    // several threads push concurrently to it

    pthread_attr_t attr;
    pthread_t thread[NB_THREADS];
    pthread_mutexattr_t mutex_attr;
    int i;

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_mutexattr_init(&mutex_attr);

    thread_test_push_args_t args[NB_THREADS];
    for (i = 0; i < NB_THREADS; i++)
    {
        args[i].id  = i;
        pthread_create(&thread[i], &attr, &thread_measure_stack_push, &args[i]);
    }


    for (i = 0; i < NB_THREADS; i++)
    {
        pthread_join(thread[i], NULL);
    }
    return;
}


// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 3

void*
thread_one(void* arg)
{
    stack_t *element = (stack_t*) malloc(sizeof(stack_t));
    printf("thread one attempting pop\n");
    stack_pop_aba(&stack, element);
    printf("thread one popped: %c\n", *((char*)(element->data)));
    printf("thread one attempting pop\n");
    return NULL;
}

void*
thread_two(void* arg)
{
    stack_t* first_item = NULL, *second_item = NULL;
    
    lock_aba_lock(2);
    unlock_aba_lock(2);
    lock_aba_lock(1);
    unlock_aba_lock(1);

    stack_pop(&stack, first_item);
    //printf("thread two popped: %c\n", *((char*)(first_item->data)));
    stack_pop(&stack, second_item);
    //printf("thread two popped: %c\n", *((char*)(second_item->data)));
    stack_push(&stack, arg);
    printf("thread two pushed a\n");
    return NULL;
}

void
print_stack()
{
    stack_t* tmp_stack = stack;
    printf("Stack: ");
    while (tmp_stack->next != NULL)
    {
	printf("%c ", *((char*)tmp_stack->data));
	tmp_stack = tmp_stack->next;
    }
    printf("\n");
}

int
test_aba()
{
#if NON_BLOCKING == 2
    pthread_t thread[ABA_NB_THREADS];

    int success, aba_detected = 0;
    // Write here a test for the ABA problem
    stack_t* a_item = malloc(sizeof(stack_t));
    stack_t* b_item = malloc(sizeof(stack_t));
    stack_t* c_item = malloc(sizeof(stack_t));
    char a,b,c;
    stack_t* popped;

    a = 'a';
    b = 'b';
    c = 'c';
    a_item->data = &a;
    b_item->data = &b;
    c_item->data = &c;
    stack_push(&stack, c_item);
    stack_push(&stack, b_item);
    stack_push(&stack, a_item);
    
    print_stack();
    
    lock_aba_lock(0);
    lock_aba_lock(2);
    pthread_create(&thread[0], NULL, &thread_one, &popped);
    while(trylock_aba_lock(1) == 0){
	unlock_aba_lock(1);
    }
    
    //   pthread_create(&thread[1], NULL, &thread_two, &a_item);    

    unlock_aba_lock(2);
    //pthread_join(thread[1], NULL);
    
    unlock_aba_lock(0);

    pthread_join(thread[0], NULL);
    
    print_stack();

    success = aba_detected;
    return success;
#else
    return false;
#endif
}

// We test here the CAS function
struct thread_test_cas_args
{
    int id;
    size_t *counter;
    pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void *
thread_test_cas(void *arg)
{
    thread_test_cas_args_t *args = (thread_test_cas_args_t *) arg;
    int i;
    size_t old, local;

    for (i = 0; i < MAX_PUSH_POP; i++)
    {
        do
        {
            old = *args->counter;
            local = old + 1;
        }
        while (cas(args->counter, old, local) != old);
    }

    return NULL;
}

int
test_cas()
{
#if 1
    pthread_attr_t attr;
    pthread_t thread[NB_THREADS];
    thread_test_cas_args_t args[NB_THREADS];
    pthread_mutexattr_t mutex_attr;
    pthread_mutex_t lock;

    size_t counter;

    int i, success;

    counter = 0;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutex_init(&lock, &mutex_attr);

    for (i = 0; i < NB_THREADS; i++)
    {
        args[i].id = i;
        args[i].counter = &counter;
        args[i].lock = &lock;
        pthread_create(&thread[i], &attr, &thread_test_cas, (void *) &args[i]);
    }

    for (i = 0; i < NB_THREADS; i++)
    {
        pthread_join(thread[i], NULL);
    }

    success = counter == (size_t)(NB_THREADS * MAX_PUSH_POP);

    if (!success)
    {
        printf("Got %ti, expected %i\n", counter, NB_THREADS * MAX_PUSH_POP);
    }

    assert(success);

    return success;
#else
    int a, b, c, *a_p, res;
    a = 1;
    b = 2;
    c = 3;

    a_p = &a;

    printf("&a=%X, a=%d, &b=%X, b=%d, &c=%X, c=%d, a_p=%X, *a_p=%d; cas returned %d\n", (unsigned int)&a, a, (unsigned int)&b, b, (unsigned int)&c, c, (unsigned int)a_p, *a_p, (unsigned int) res);

    res = cas((void **)&a_p, (void *)&c, (void *)&b);

    printf("&a=%X, a=%d, &b=%X, b=%d, &c=%X, c=%d, a_p=%X, *a_p=%d; cas returned %X\n", (unsigned int)&a, a, (unsigned int)&b, b, (unsigned int)&c, c, (unsigned int)a_p, *a_p, (unsigned int)res);

    return 0;
#endif
}

// Stack performance test
#if MEASURE != 0
struct stack_measure_arg
{
    int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;
#endif

int
main(int argc, char **argv)
{
    setbuf(stdout, NULL);
    // MEASURE == 0 -> run unit tests
    test_init();
#if MEASURE == 0


    test_run(test_cas);

    test_run(test_push_safe);
    test_run(test_pop_safe);
    test_run(test_aba);

    test_finalize();
#else
    // Run performance tests
    int i;
    stack_measure_arg_t arg[NB_THREADS];

    test_setup();

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (i = 0; i < NB_THREADS; i++)
    {
        arg[i].id = i;
        (void)arg[i].id; // Makes the compiler to shut up about unused variable arg
        // Run push-based performance test based on MEASURE token
#if MEASURE == 1
	
        clock_gettime(CLOCK_MONOTONIC, &t_start[i]);
	measure_push();
        // Push MAX_PUSH_POP times in parallel
        clock_gettime(CLOCK_MONOTONIC, &t_stop[i]);
#else
	measure_push();
        // Run pop-based performance test based on MEASURE token
        clock_gettime(CLOCK_MONOTONIC, &t_start[i]);
        // Pop MAX_PUSH_POP times in parallel
	measure_pop();
        clock_gettime(CLOCK_MONOTONIC, &t_stop[i]);
#endif
    }

    // Wait for all threads to finish
    clock_gettime(CLOCK_MONOTONIC, &stop);

    // Print out results
    for (i = 0; i < NB_THREADS; i++)
    {
        printf("%i %i %li %i %li %i %li %i %li\n", i, (int) start.tv_sec,
               start.tv_nsec, (int) stop.tv_sec, stop.tv_nsec,
               (int) t_start[i].tv_sec, t_start[i].tv_nsec, (int) t_stop[i].tv_sec,
               t_stop[i].tv_nsec);
    }
#endif

    return 0;
}
