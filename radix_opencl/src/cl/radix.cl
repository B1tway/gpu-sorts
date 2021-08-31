#ifndef WORK_GROUP_SIZE
    #define WORK_GROUP_SIZE 128
#endif

#ifndef DIGITS_PER_STEP
    #define DIGITS_PER_STEP 2
    #ifndef VALUES_PER_DIGIT
        #define VALUES_PER_DIGIT (1 << DIGITS_PER_STEP)
    #endif 
#endif

#ifndef VALUES_PER_DIGIT
    #define VALUES_PER_DIGIT 4
#endif 

__kernel void local_sum(__global unsigned int* in,
                        __global unsigned int* indexes,
                        __global unsigned int* sums,
                        unsigned int original_mask, 
                        unsigned int step, 
                        unsigned int length) {

    unsigned int mask = (original_mask << step);
    unsigned int gid = get_global_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int group_size = get_local_size(0);
    unsigned int max_local_id = min(group_size, length - group_size * group_id);

    unsigned int local_arrays_dim = (VALUES_PER_DIGIT - 1);

    __local unsigned int local_as[WORK_GROUP_SIZE];

   
    __local unsigned int local_sums[WORK_GROUP_SIZE * (VALUES_PER_DIGIT - 1)];
    unsigned int buffers[VALUES_PER_DIGIT - 1];

    if (gid < length) {
        local_as[lid] = in[gid];

        unsigned int masked = (local_as[lid] & mask) >> step;

        if (lid < max_local_id - 1) {
           
            for (int i = 0; i < local_arrays_dim; ++i) {
                local_sums[lid + 1 + i * group_size] = 0;
            }
           
            if (masked ^ original_mask) {
                local_sums[lid + 1 + masked * group_size] = 1;
            }
        } else {
            for (int i = 0; i < local_arrays_dim; ++i) {
                local_sums[i * group_size] = 0;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    
    for (unsigned int k = 1; k < group_size; k *= 2) {
        if (lid < max_local_id && lid >= k) {
            for (int i = 0; i < local_arrays_dim; ++i) {
                buffers[i] = local_sums[lid - k + i * group_size] + local_sums[lid + i * group_size];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid < max_local_id && lid >= k) {
            for (int i = 0; i < local_arrays_dim; ++i) {
                local_sums[lid + i * group_size] = buffers[i];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    
    if (lid == 0) {
        unsigned int masked = (local_as[max_local_id - 1] & mask) >> step;

        unsigned int last_index = VALUES_PER_DIGIT * (group_id + 1);

        sums[last_index + local_arrays_dim] = max_local_id;

        for (int i = 0; i < local_arrays_dim; ++i) {
            unsigned int result = local_sums[max_local_id - 1 + i * group_size];
            sums[last_index + i] = result;
            sums[last_index + local_arrays_dim] -= result;
        }

       
        if (masked ^ original_mask) {
            sums[last_index + masked] += 1;
            sums[last_index + local_arrays_dim] -= 1;
        }
    }

  
    if (gid < length) {
        unsigned int masked = (local_as[lid] & mask) >> step;

        unsigned int local_index;

        if (masked ^ original_mask) {
            local_index = local_sums[lid + masked * group_size];
        } else {
            local_index = lid;
            for (int i = 0; i < local_arrays_dim; ++i) {
                local_index -= local_sums[lid + i * group_size];
            }
        }

        indexes[gid] = local_index;
    }
}

__kernel void radix(__global unsigned int* in,
                    __global unsigned int* as_result,
                    __global unsigned int* indexes,
                    __global unsigned int* sums,
                    unsigned int original_mask, 
                    unsigned int step, 
                    unsigned int length) {

    unsigned int mask = original_mask << step;
    unsigned int gid = get_global_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int group_count = get_num_groups(0);
    unsigned int last_index = VALUES_PER_DIGIT * group_count;

  
    if (gid < length) {
        unsigned int value = in[gid];
        unsigned int masked = (value & mask) >> step;

        unsigned int new_index = indexes[gid] + sums[VALUES_PER_DIGIT * group_id + masked];

        for (int i = 0; i < masked; ++i) {
            new_index += sums[last_index + i];
        }

        as_result[new_index] = value;
    }
}
