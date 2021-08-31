#include <string.h>
#include <iostream>
#include <fstream>

#include "dispatch.hpp"
#include "op_params.hpp"

struct sort_data {
    uint64_t in;
    uint64_t out;
    uint32_t length;
    uint32_t current_length;
};

bool read_file(const char* path, vector<char>& bin)
{
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	const auto size = file.tellg();
	bool read_failed = false;
	do {
		if (size < 0) { read_failed = true; break; }
		    file.seekg(std::ios::beg);
		if (file.fail()) { read_failed = true; break; }
		    bin.resize(size);
		if (file.rdbuf()->sgetn(bin.data(), size) != size) { read_failed = true; break; }
	} while (false);
	file.close();

	if (read_failed)
		cerr << "unable to read file \"" << path << "\"\n";

	return !read_failed;
}

bool write_file(const char* path, vector<char>& bin)
{
	std::ofstream file(path, std::ios::binary | std::ios::out);
	file.write(bin.data(), bin.size());
	file.close();

    auto write_ok = file.good();

	if (write_ok)
		cerr << "unable to write file \"" << path << "\"\n";

	return write_ok;
}

using namespace std;
using namespace amd::dispatch;

class VectorMergeSort : public Dispatch
{
private:
    unsigned length;
    unsigned current_length;
    int gpu_id;
    int dbg_size;
    string dbg_path;
    string co_path;
    string build_cmd;
    kernel kern;
    vector<float> in;
    vector<float> out;
    void *in_gpu_ptr;
    void *out_gpu_ptr;
    void *dbg_buf_ptr;
   
public:
    VectorMergeSort(string &co_path, string &build_cmd, int gpu_id, int dbg_size, string &dbg_path, unsigned int lenght)
        : Dispatch(),
            length(lenght),
            gpu_id(gpu_id),
            dbg_size(dbg_size),
            dbg_path{std::move(dbg_path)},
            co_path{std::move(co_path)},
            build_cmd{std::move(build_cmd)},
            in_gpu_ptr{nullptr},
            current_length{1},
            out_gpu_ptr{nullptr}, 
            dbg_buf_ptr{nullptr} { }

    bool Init()
    {
        if (!init(gpu_id))
            return false;

        // debug buffer must be allocated before the shader build
        // as debug scripts use ASM_DBG_BUF_ADDR env variable
        if (dbg_size > 0)
        {
            stringstream ss;
            dbg_buf_ptr = allocate_gpumem(dbg_size);

            ss << dbg_buf_ptr;
            auto dbg_ptr_str = ss.str();

            ss.str(""); // clear stream
            ss << dbg_size;
            auto dbg_sze_str = ss.str();

            if ((setenv("ASM_DBG_BUF_ADDR", dbg_ptr_str.c_str(), 1) != 0)
            ||  (setenv("ASM_DBG_BUF_SIZE", dbg_sze_str.c_str(), 1) != 0))
            { 
                cerr << "Failed setup ASM_DBG_BUF_ADDR ASM_DBG_BUF_SIZE env variables\n";
                return false;
            }
        }
        cout << "Length: " << length << std::endl;
        cout << "Execute: " << build_cmd << std::endl;
        if (system(build_cmd.c_str()))
        {
            cerr << "Error: assembly failed \n";
            return false;
        }

        vector<char> bin;
        if (!read_file(co_path.c_str(), bin))
            return false;

        return load_kernel_from_memory(&kern, bin.data(), bin.size());
    }

    bool Setup()
    {
        in.resize(length);
        out.resize(length);

        for (unsigned i = 0; i < length; i++) {
            in[i] = (float) (length - i);
        }

        auto buf_size = length * sizeof(float);
        in_gpu_ptr = allocate_gpumem(buf_size);

        if (!(memcpyHtoD(in_gpu_ptr, in.data(), buf_size))) {
            cerr << "Error: failed to copy to local" << std::endl;
            return false;
        }

        out_gpu_ptr = allocate_gpumem(buf_size);
        sort_data app_params;
        app_params.in = (uint64_t)  in_gpu_ptr;
        app_params.out = (uint64_t) out_gpu_ptr;
        app_params.length = length;
        app_params.current_length = current_length;
        dispatch_params params;
        params.wg_size[0] = 128;
        params.wg_size[1] = 1;
        params.wg_size[2] = 1;
        std::cout << "Workgroup size " << "[" << params.wg_size[0] << ", " << params.wg_size[1] << ", "  << params.wg_size[2] << "]\n"; 
        params.grid_size[0] = (length +  params.wg_size[0] - 1) /  params.wg_size[0] *  params.wg_size[0];
        params.grid_size[1] = params.wg_size[1];
        params.grid_size[2] = params.wg_size[2];
        std::cout << "Grid size " << "[" << params.grid_size[0] << ", " << params.grid_size[1] << ", "  << params.grid_size[2] << "]\n";
        params.dynamic_lds = 0; 
        params.kernarg_size = sizeof(sort_data);
        params.kernarg = &app_params;
        bool res = true;
        while(app_params.current_length < length)
        {
            std::cout << "Run kernel with the current_length = "<< app_params.current_length << "\n";
            res &= run_kernel(&kern, &params, UINT64_MAX);
            app_params.current_length *= 2;
            std::swap(app_params.in , app_params.out);
        }
        return res;
    }

    bool Verify()
    {
        double exec_time = sum_clocks/((double) CLOCKS_PER_SEC) * 1'000'000;
        std::cout << "Execution time: " <<  exec_time << " microsec" << std::endl;
        if (!memcpyDtoH(out.data(), out_gpu_ptr, out.size() * sizeof(float)))
        {
            cerr << "Error: failed to copy from local" << std::endl;
            return false;
        }
         if (!memcpyDtoH(in.data(), in_gpu_ptr, in.size() * sizeof(float)))
        {
            cerr << "Error: failed to copy from local" << std::endl;
            return false;
        }
        bool ok = true;
        for (unsigned i = 0; i < length; ++i)
        {   
            float res = out[i];
            if(in[0] == 1) {
                res = in[i];
            }
            
           // std::cout << in[i] << " " << out[i] << std::endl;
            float expected = i + 1;

            if (expected != res)
            {
                cerr << "Error: validation failed at " << i << ": got " << res << " expected " << expected << std::endl;
                ok = false;
            }
        }
        return ok;
    }

    bool Shutdown()
    {
        if (in_gpu_ptr != nullptr)
        {
            if (!free_gpumem(in_gpu_ptr))
                cerr << "Failed free in buff\n";
        }
        
        if (out_gpu_ptr != nullptr)
        {
            if (!free_gpumem(out_gpu_ptr))
                cerr << "Failed free out buff\n";
        }


        if (dbg_buf_ptr != 0)
        {
            vector<char> bin(dbg_size);

            if (!(memcpyDtoH(bin.data(), dbg_buf_ptr, bin.size()) 
                && free_gpumem(dbg_buf_ptr) 
                && write_file(dbg_path.c_str(), bin)))
                cerr << "Error: failed to copy debug buffer to host\n";
        }

        return shutdown();
    }

    bool Run()
    {
        auto ok = Init()
               && Setup()
               && Verify();

        ok = Shutdown() && ok;
        
        cout << (ok ? "Success" : "Failed") << endl;
        return ok;
    }
};

int main(int argc, const char **argv)
{
    int gpu_id;
    std::string co_path;
    std::string build_cmd;
    int dbg_size;
    std::string dbg_path;
     unsigned int length;
    Options cli_ops(100);

    cli_ops.Add(&gpu_id, "-gpu-id", "", 0, "GPU agent id", atoi);
    cli_ops.Add(&co_path, "-c", "", string(""), "Code object path", str2str);
    cli_ops.Add(&build_cmd, "-asm", "", string(""), "Shader build cmd", str2str);
    cli_ops.Add(&dbg_size, "-bsz", "", 0, "Debug buffer size", atoi);
    cli_ops.Add(&dbg_path, "-b", "", string(""), "Debug buffer path", str2str);
    cli_ops.Add(&dbg_path, "-b", "", string(""), "Debug buffer path", str2str);
    cli_ops.Add(&length,  "-l", "", 64u, "vector length", str2u);
    for (int i = 1; i <= argc - 1; i += 2)
    {
        if (!strcmp(argv[i], "-?") || !strcmp(argv[i], "-help"))
        {
            cli_ops.ShowHelp();
            exit(0);
            return false;
        }

        bool merged_flag = false;
        if (!cli_ops.ProcessArg(argv[i], argv[i + 1], &merged_flag))
        {
            std::cerr << "Unknown flag or flag without value: " << argv[i] << "\n";
            return false;
        }

        if (merged_flag)
        {
            i--;
            continue;
        }

        if (argv[i + 1] && cli_ops.MatchArg(argv[i + 1]))
        {
            std::cerr << "Argument \"" << argv[i + 1]
                      << "\" is aliased with command line flags\n\t maybe real argument is missed for flag \""
                      << argv[i] << "\"\n";
            return false;
        }
    }

    return VectorMergeSort(co_path, build_cmd, gpu_id, dbg_size, dbg_path, length).Run();
}
