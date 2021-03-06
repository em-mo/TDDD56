/*
 * stack.h
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

#include <stdlib.h>
#include <pthread.h>

#ifndef STACK_H
#define STACK_H

struct stack
{
  // This is a fake structure; change it to your needs
    void* data;
    struct stack* next;
};

typedef struct stack stack_t;

// Pushes an element in a thread-safe manner
int stack_push(stack_t **, void*);
// Pops an element in a thread-safe manner
int stack_pop(stack_t **, void*);

int
stack_init(stack_t*, size_t);

stack_t * stack_alloc();


int
stack_pop_aba(stack_t **, void *);

void
lock_aba_lock(int lock_id);

void
unlock_aba_lock(int lock_id);

int
trylock_aba_lock(int lock_id);

#endif /* STACK_H */
