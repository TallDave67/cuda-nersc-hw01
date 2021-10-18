# Homework 1

These exercises will have you write some basic CUDA programs. You will learn how to allocate GPU memory, move data between the host and the GPU, and launch kernels.

## **A. Hello World**

Your first task is to complete the skeleton of the simple hello world program. 

Note the use of `cudaDeviceSynchronize()` after the kernel launch. In CUDA, kernel launches are *asynchronous* to the host thread. The host thread will launch a kernel but not wait for it to finish, before proceeding with the next line of host code. Therefore, to prevent application termination before the kernel gets to print out its message, we must use this synchronization function.

## **B. Vector Add**

Your second task is to complete the skeleton of the vector add program.

## **C. Matrix Multiply (naive)**

Your third task is to complete the skeleton of the naive matrix multiply program. 

This example introduces 2D threadblock/grid indexing, something we did not cover in lesson 1. If you study the code you will probably be able to see how it is a structural extension from the 1D case.

This code includes built-in error checking, so a correct result is indicated by the program.
