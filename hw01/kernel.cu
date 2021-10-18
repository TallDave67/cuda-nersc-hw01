
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int main_hello();
int main_vector_add();
int main_matrix_mul_float();
int main_matrix_mul_double();

int main()
{
    return main_hello();
    //return main_vector_add();
    //return main_matrix_mul_float();
    //return main_matrix_mul_double();
}
