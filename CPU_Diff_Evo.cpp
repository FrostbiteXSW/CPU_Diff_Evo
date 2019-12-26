#include <stdio.h>
#include <chrono>
#include <mpi.h>
#include <omp.h>

using namespace std;
using namespace chrono;

#define NUM_THREADS 16		// CPU线程数
#define ITERATIONS  10000	// 差分进化次数
#define ARG_NUM	    10		// 参数个数
#define ARG_LIMIT   100		// 参数绝对值范围限制
#define BIAS	    22		// 偏差
#define F		    0.5		// 缩放因子
#define CR		    0.3		// 交叉概率

enum tags {
	send_arg_list,
	recv_rst_list
};

/*
 * 解方程形如：
 * 
 *		∑(-1)^i * (x_i)^i * i  + BIAS = 0, i > 0
 *		
 *	其中，x_i的绝对值小于等于ARG_LIMIT
 *	ARG_NUM决定变量的数量（即i）
 */


/**
 * \brief 单进程进行差分进化，计算子代的参数。
 * \param arg_list 当前最优参数列表
 * \param result_list 进程中所有线程计算结果的最优子代。
 */
void DifferentialEvolution(const double* arg_list, double* result_list) {
	// 计算结果缓冲区
	const auto results = new double*[NUM_THREADS];
	for (auto tid = 0; tid < NUM_THREADS; ++tid) {
		results[tid] = new double[ARG_NUM + 1];
	}
	
	// 变异
#pragma omp parallel for shared(results)
	for (auto tid = 0; tid < NUM_THREADS; ++tid) {
		for (auto i = 0; i < ARG_NUM; ++i) {
			int r1, r2, r3;
			do {
				r1 = int(double(rand()) / RAND_MAX * ARG_NUM) % ARG_NUM;
				r2 = int(double(rand()) / RAND_MAX * ARG_NUM) % ARG_NUM;
				r3 = int(double(rand()) / RAND_MAX * ARG_NUM) % ARG_NUM;
			}
			while (r1 == r2 || r2 == r3 || r1 == r3);
			results[tid][i] = arg_list[r1] + F * (arg_list[r2] - arg_list[r3]);
			if (abs(results[tid][i]) > ARG_LIMIT) {
				results[tid][i] = (double(rand()) / RAND_MAX - 0.5) * 2 * ARG_LIMIT;
			}
		}
	}

	// 交叉
#pragma omp parallel for shared(results)
	for (auto tid = 0; tid < NUM_THREADS; ++tid) {
		const auto j = int(double(rand()) / RAND_MAX * ARG_NUM) % ARG_NUM;
		for (auto i = 0; i < ARG_NUM; ++i) {
			if (i != j && double(rand()) / RAND_MAX > CR) {
				results[tid][i] = arg_list[i];
			}
		}
	}

	// 计算
#pragma omp parallel for shared(results)
	for (auto tid = 0; tid < NUM_THREADS; ++tid) {
		results[tid][ARG_NUM] = BIAS;
		for (auto i = 0; i < ARG_NUM; ++i) {
			results[tid][ARG_NUM] += (i + 1.) * pow(-1, i + 1.) * pow(results[tid][i], i + 1.);
		}
	}

	// 选择
	auto bestResultIndex = 0;
	for (auto tid = 1; tid < NUM_THREADS; ++tid) {
		if (abs(results[tid][ARG_NUM]) < abs(results[bestResultIndex][ARG_NUM])) {
			bestResultIndex = tid;
		}
	}

	// 拷贝结果
	memcpy(result_list, results[bestResultIndex], sizeof(double) * (ARG_NUM + 1));

	// 释放内存
#pragma omp parallel for shared(results)	
	for (auto tid = 1; tid < NUM_THREADS; ++tid) {
		delete[] results[tid];
	}
	delete[] results;
}

void MainProc(const int proc_cnt) {
	MPI_Status status;
	
	// 输出每进程的线程数
	printf("Threads per process: %d\n", omp_get_max_threads());

	// 当前最优参数列表及其结果（[argv], result）
	const auto argList = new double[ARG_NUM + 1];

	// 各进程计算得到的最优子代参数及其结果
	const auto resultList = new double*[proc_cnt];
	for (auto i = 0; i < proc_cnt; ++i) {
		resultList[i] = new double[ARG_NUM + 1];
	}

	// 初始化种群
	const auto seed = time(nullptr);
	srand(seed);
	for (auto i = 0; i < ARG_NUM; ++i) {
		argList[i] = (double(rand()) / RAND_MAX - 0.5) * 2 * ARG_LIMIT;
	}
	argList[ARG_NUM] = BIAS;
	for (auto i = 0; i < ARG_NUM; ++i) {
		argList[ARG_NUM] += (i + 1.) * pow(-1, i + 1.) * pow(argList[i], i + 1.);
	}

	// 差分进化	
	const auto start = system_clock::now();
	for (auto i = 0; i < ITERATIONS; ++i) {
		// 主进程发送父代参数给子进程
		for (auto pid = 0; pid < proc_cnt - 1; ++pid) {
			MPI_Send(argList, ARG_NUM + 1, MPI_DOUBLE, pid, send_arg_list, MPI_COMM_WORLD);
		}

		// 主进程计算最优子代结果
		DifferentialEvolution(argList, resultList[proc_cnt - 1]);

		// 接收子进程计算结果
		for (auto pid = 0; pid < proc_cnt - 1; ++pid) {
			MPI_Recv(resultList[pid], ARG_NUM + 1, MPI_DOUBLE, pid, recv_rst_list, MPI_COMM_WORLD, &status);
		}

		// 主进程进行子代选择
		auto bestResultIndex = -1;
		for (auto j = 0; j < proc_cnt; ++j) {
			if (abs(resultList[j][ARG_NUM]) < abs(argList[ARG_NUM])) {
				bestResultIndex = j;
			}
		}
		if (bestResultIndex >= 0) {
			memcpy(argList, resultList[bestResultIndex], sizeof(double) * (ARG_NUM + 1));
		}
	}
	const auto elapsedTime = duration_cast<milliseconds>(system_clock::now() - start).count();
	printf("Algorithm running time is %lld ms\n", elapsedTime);

	// 输出结果
	for (auto i = 0; i < ARG_NUM; ++i) {
		printf("x%d = %f\n", i + 1, argList[i]);
	}
	printf("Result = %f\n", argList[ARG_NUM]);

	// 测试结果
	double realResult = BIAS;
	for (auto i = 0; i < ARG_NUM; ++i) {
		realResult += pow(-1, i + 1.) * pow(argList[i], i + 1.) * (i + 1.);
	}
	printf("Validating Result = %f\n", realResult);

	// 释放内存
	delete[] argList;
	for (auto i = 0; i < proc_cnt; ++i) {
		delete[] resultList[i];
	}
	delete[] resultList;
}

void SubProc(const int proc_cnt) {
	MPI_Status status;

	// 初始化随机数因子
	const auto seed = time(nullptr);
	srand(seed);

	// 当前最优参数列表及其结果（[argv], result）
	const auto argList = new double[ARG_NUM + 1];

	// 当前进程计算得到的最优子代参数及其结果
	const auto resultList = new double[ARG_NUM + 1];

	// 差分进化
	for (auto i = 0; i < ITERATIONS; ++i) {
		// 接收父代参数
		MPI_Recv(argList, ARG_NUM + 1, MPI_DOUBLE, proc_cnt - 1, send_arg_list, MPI_COMM_WORLD, &status);

		// 计算最优子代结果
		DifferentialEvolution(argList, resultList);

		// 发送最优子代结果
		MPI_Send(resultList, ARG_NUM + 1, MPI_DOUBLE, proc_cnt - 1, recv_rst_list, MPI_COMM_WORLD);
	}

	// 释放内存
	delete[] argList;
	delete[] resultList;
}

int main(int argc, char* argv[]) {
	int procCnt, procId;

	// 初始化MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &procId);
	MPI_Comm_size(MPI_COMM_WORLD, &procCnt);

	// 设置线程总数
	omp_set_num_threads(NUM_THREADS);

	if (procId == procCnt - 1) {
		// 最后一个进程为主进程
		MainProc(procCnt);
	}
	else {
		// 其他进程为子进程
		SubProc(procCnt);
	}

	MPI_Finalize();
}
