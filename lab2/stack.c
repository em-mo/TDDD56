/*
 * stack.c
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
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
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

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through lock-based CAS
#else
#warning Stacks are synchronized through hardware CAS
#endif
#endif

#if NON_BLOCKING == 0
pthread_mutex_t stack_mutex = PTHREAD_MUTEX_INITIALIZER;
#else
pthread_mutex_t aba_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

stack_t *
stack_alloc()
{
    // Example of a task allocation with correctness control
    // Feel free to change it
    stack_t *res;

    res = malloc(sizeof(stack_t));
    assert(res != NULL);

    if (res == NULL)
        return NULL;

    // You may allocate a lock-based or CAS based stack in
    // different manners if you need so
#if NON_BLOCKING == 0
    // Implement a lock_based stack
#elif NON_BLOCKING == 1
    /*** Optional ***/
    // Implement a harware CAS-based stack
#else
    // Implement a harware CAS-based stack
#endif

    return res;
}

int
stack_init(stack_t *stack, size_t size)
{
    assert(stack != NULL);
    assert(size > 0);

    stack->next = NULL;
    stack->data = NULL;

#if NON_BLOCKING == 0
    // Implement a lock_based stack
//    pthread_mutex_init(&stack_mutex, NULL);
#elif NON_BLOCKING == 1
    /*** Optional ***/
    // Implement a harware CAS-based stack
#else
    // Implement a harware CAS-based stack
#endif

    return 0;
}

int
stack_check(stack_t *stack)
{
    /*** Optional ***/
    // Use code and assertions to make sure your stack is
    // in a consistent state and is safe to use.
    //
    // For now, just makes just the pointer is not NULL
    //
    // Debugging use only

    assert(stack != NULL);

    return 0;
}
int
stack_push(stack_t **stack, void *buffer)
{
#if NON_BLOCKING == 0
    // Implement a lock_based stack
    pthread_mutex_lock(&stack_mutex);

    ((stack_t *)buffer)->next = *stack;
    *stack = (stack_t *) buffer;

    pthread_mutex_unlock(&stack_mutex);
#elif NON_BLOCKING == 1
    /*** Optional ***/
    // Implement a harware CAS-based stack
#else
    // Implement a harware CAS-based stack

    //WIP
    /*    stack_t *old_item;
        do{
        old_item = stack->next;
        new_item->next = stack->next;
        cas(stack->next, old_item, new_item);
        }while(new_item != old_item)*/

    stack_t *old_item;
    do {
        old_item = *stack;
        ((stack_t *)buffer)->next = old_item;
        
    } while (cas(stack, old_item, buffer) != old_item);
#endif

    return 0;
}

int
stack_pop(stack_t **stack, void *buffer)
{
#if NON_BLOCKING == 0
    // Implement a lock_based stack
    pthread_mutex_lock(&stack_mutex);

    buffer = *stack;
    *stack = (*stack)->next;

    pthread_mutex_unlock(&stack_mutex);

#elif NON_BLOCKING == 1
    /*** Optional ***/
    // Implement a harware CAS-based stack
#else
    stack_t* next;
    do {
        buffer = *stack;
        next = (*stack)->next;
        
    } while (cas(stack, buffer, next) != buffer);
#endif

    return 0;
}

#if NON_BLOCKING == 2
void
lock_aba_lock()
{
     pthread_mutex_lock(&aba_mutex);
     return;
}

void
unlock_aba_lock()
{
     pthread_mutex_unlock(&aba_mutex);
     return;
}

int
stack_pop_aba(stack_t **stack, void *buffer)
{
    stack_t* next;
    do {
        buffer = *stack;
        next = (*stack)->next;
        pthread_mutex_lock(&aba_mutex);        
    } while (cas(stack, buffer, next) != buffer);

    return 0;
}
#endif
