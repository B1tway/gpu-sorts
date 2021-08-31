#define WORK_GROUP_SIZE 256

__kernel void bitonic_local(__global float* in, unsigned int batch_size, unsigned int size, unsigned int length) {
	unsigned int local_id = get_local_id(0);
	unsigned int gid = get_global_id(0);

	__local float batch[WORK_GROUP_SIZE];

	if (gid < length) {
		batch[local_id] = in[gid];
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	int tone_flag = gid % (2 * size) < size;

	while (batch_size >= 1) {

		if (gid % (2 * batch_size) < batch_size && gid + batch_size < length) {
			float a = batch[local_id];
			float b = batch[local_id + batch_size];
			
			if ((a > b) == tone_flag) {
				batch[local_id] = b;
				batch[local_id + batch_size] = a;
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		batch_size >>= 1;
	}

	if (gid < length) {
		in[gid] = batch[local_id];
	}
}


__kernel void bitonic(__global float* in, unsigned int batch_size, unsigned int size, unsigned int length) {
	unsigned int gid = get_global_id(0);

	int tone_flag = gid % (2 * size) < size;

	if (gid % (2 * batch_size) < batch_size && gid + batch_size < length) {
		float a = in[gid];
		float b = in[gid + batch_size];
		if ((a > b) == tone_flag) {
			in[gid] = b;
			in[gid + batch_size] = a;
		}
	}
}
