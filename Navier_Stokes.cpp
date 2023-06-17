#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <mpi.h>

using namespace std;
#define DO_TIMING
#define EXCLUDE_FILE_WRITE
int Nx = 201;
int Ny = 101;
double Lx = 0.1, Ly = 0.05;
double rho = 1000, nu = 1e-6;
double P_max = 0.5;
// const double t_end = 50.0;
double t_end = 10.0;
double dt_min = 1.e-3;
double courant = 0.01;
double dt_out = 0.5;

// this class provide the simulation functionalities to solve NS function in parallel
class Simulator
{
public:
	// p_count: total process count
	// id: process id
	// dim_x: number of subdomains in x direction
	// dim_y: number of subdomains in y direction
	// idx_x: starting x index of the domain without boundary
	// idx_y: starting y index of the domain without boundary
	// len_x: number of interior points in x direction
	// len_y: number of interior points in y direction
	int p_count, id, dim_x, dim_y, idx_x, idx_y, len_x, len_y;
	// my_idx_x: x index of the subdomain in the decomposition
	// my_idx_y: y index of the subdomain in the decomposition
	int my_idx_x, my_idx_y;
	// communication variables
	int req_cnt, tag_num;
	MPI_Request *req_list;

	// P, P_old: pressure
	// u, u_old: velocity
	// v, v_old: velocity
	// PPrhs: laplace of pressure
	double *P, *P_old, *u, *u_old, *v, *v_old, *PPrhs;
	// Datatype_row: MPI datatype for sending a continues row
	// Datatype_col: MPI datatype for sending a column with stride in memory
	MPI_Datatype Datatype_row, Datatype_col;
	// id of the processes that need to communicate with this process, -1 as boundary
	// [left, up, right, down]
	int neighbors[4];
	// domain coeffcients
	double dx, dy, dt, t;
	Simulator();
	void grids_to_file(int out);
	void setup();
	void calculate_ppm_RHS_central();
	void solve_NS(void);
	void set_pressure_BCs(void);
	int pressure_poisson_jacobi(double rtol);
	void calculate_intermediate_velocity(int time_it);
	void set_velocity_BCs(void);
	double project_velocity(void);
	void find_dimensions();
	int get_idx(int x, int y);
	void build_row_type();
	void build_col_type();
	void sendhelper(double *tosend, int &cnt);
	void write_config();
	void read_config_file(char *argv);

	~Simulator();
};

// constructor, setup the MPI and find the decomposition dimensions
Simulator::Simulator()
{
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p_count);
	find_dimensions();
}
// destructor, clean up the memory and free MPI types
Simulator::~Simulator()
{
	MPI_Type_free(&Datatype_row);
	MPI_Type_free(&Datatype_col);

	delete[] P;
	delete[] P_old;
	delete[] u;
	delete[] u_old;
	delete[] v;
	delete[] v_old;
	delete[] PPrhs;
	delete[] req_list;
	MPI_Finalize();

}
void Simulator::read_config_file(char *argv)
{
	if (id == 0)
	{
		fstream newfile;
		// open a file to perform read operation using file object
		newfile.open(argv, ios::in); 
		// checking whether the file is open
		if (newfile.is_open())
		{ 
			string tp;
			while (getline(newfile, tp))
			{
				int end = tp.find(' ');
				string v_name = tp.substr(0, end);
				if (v_name.compare("Nx") == 0)
				{
					cout << stoi(&tp[end]) << endl;
					Nx = stoi(&tp[end]);
				}
				if (v_name.compare("Ny") == 0)
				{
					Ny = stoi(&tp[end + 1]);
				}
				if (v_name.compare("Lx") == 0)
				{
					Lx = stod(&tp[end + 1]);
				}
				if (v_name.compare("Ly") == 0)
				{
					Ly = stod(&tp[end + 1]);
				}
				if (v_name.compare("rho") == 0)
				{
					rho = stod(&tp[end + 1]);
				}
				if (v_name.compare("nu") == 0)
				{
					nu = stod(&tp[end + 1]);
				}
				if (v_name.compare("P_max") == 0)
				{
					P_max = stod(&tp[end + 1]);
				}
				if (v_name.compare("t_end") == 0)
				{
					t_end = stod(&tp[end + 1]);
				}
				if (v_name.compare("dt_min") == 0)
				{
					dt_min = stod(&tp[end + 1]);
				}
				if (v_name.compare("courant") == 0)
				{
					courant = stod(&tp[end + 1]);
				}
				if (v_name.compare("dt_out") == 0)
				{
					dt_out = stod(&tp[end + 1]);
				}
			}
#ifndef DO_TIMING
			cout << "Using following configurations" << endl;
			cout << "Nx: " << Nx << endl;
			cout << "Ny: " << Ny << endl;
			cout << "Lx: " << Lx << endl;
			cout << "Ly: " << Ly << endl;
			cout << "rho: " << rho << endl;
			cout << "nu: " << nu << endl;
			cout << "P_max: " << P_max << endl;
			cout << "t_end: " << t_end << endl;
			cout << "dt_min: " << dt_min << endl;
			cout << "courant: " << courant << endl;
			cout << "dt_out: " << dt_out << endl;
#endif
			// close the file object.
			newfile.close(); 
		}
	}
	MPI_Bcast(&Nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&rho, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&P_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&t_end, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dt_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&courant, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dt_out, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// save configuration file to disk
void Simulator::write_config()
{
	stringstream fname;
	fstream f1;
	fname << "./out/config_file.dat";
	f1.open(fname.str().c_str(), ios_base::out);
	f1 << "Nx"
	   << "\t" << Nx << endl;
	f1 << "Ny"
	   << "\t" << Ny << endl;
	f1 << "Lx"
	   << "\t" << Lx << endl;
	f1 << "Ly"
	   << "\t" << Ly << endl;
	f1 << "dimx"
	   << "\t" << dim_x << endl;
	f1 << "dimy"
	   << "\t" << dim_y << endl;
	f1.close();
}
// build MPI datatype to send row, essentially a 1d array
void Simulator::build_row_type()
{
	int block_lengths[1];
	MPI_Aint offsets[1];
	MPI_Datatype typelist[1];
	typelist[0] = MPI_DOUBLE;
	block_lengths[0] = len_x;
	offsets[0] = 0;
	MPI_Type_create_struct(1, block_lengths, offsets, typelist, &Datatype_row);
	MPI_Type_commit(&Datatype_row);
}
// build MPI datatype to send col
void Simulator::build_col_type()
{
	// we need to send len_y data as we don't need the corner values
	int *block_lengths = new int[len_y];
	MPI_Aint *offsets = new MPI_Aint[len_y];
	// MPI_Aint* addresses = new MPI_Aint[len_y];
	MPI_Aint add_start;
	MPI_Datatype *typelist = new MPI_Datatype[len_y];
	MPI_Get_address(P, &add_start);
	// calculating the offsets
	for (int i = 0; i < len_y; i++)
	{
		typelist[i] = MPI_DOUBLE;
		block_lengths[i] = 1;
		MPI_Aint temp;
		MPI_Get_address(&P[i * (len_x + 2)], &temp);
		offsets[i] = temp - add_start;
	}
	// creating the datatype
	MPI_Type_create_struct(len_y, block_lengths, offsets, typelist, &Datatype_col);
	MPI_Type_commit(&Datatype_col);
	// free memroy
	delete[] block_lengths;
	delete[] offsets;
	delete[] typelist;
}
// helper function to convert 2d indexing to 1d indexing
int Simulator::get_idx(int x, int y)
{
	return y * (len_x + 2) + x;
}
// search for the decomposition dimensions
void Simulator::find_dimensions()
{
	int min_gap = p_count;
	int top = sqrt(p_count) + 1;
	for (int i = 1; i <= top; i++)
	{
		if (p_count % i == 0)
		{
			int gap = abs(p_count / i - i);

			if (gap < min_gap)
			{
				min_gap = gap;
				dim_x = p_count / i;
				dim_y = i;
				top = dim_x;
			}
		}
	}

	if (id == 0)
		cout << "Divide " << p_count << " into " << dim_x << " by " << dim_y << " grid" << endl;
}
// save result to files, each process will write an independent file and the result will be combined by
// post processing code
void Simulator::grids_to_file(int out)
{
	// Write the output for a single time step to file
	stringstream fname;
	fstream f1;
	fname << "./out/"
		  << "id_" << id << "_P"
		  << "_" << out << ".dat";
	f1.open(fname.str().c_str(), ios_base::out);
	for (int i = 0; i < len_x + 2; i++)
	{
		for (int j = 0; j < len_y + 2; j++)
			f1 << P[get_idx(i, j)] << "\t";
		f1 << endl;
	}
	f1.close();
	fname.str("");
	fname << "./out/"
		  << "id_" << id << "_u"
		  << "_" << out << ".dat";
	f1.open(fname.str().c_str(), ios_base::out);
	for (int i = 0; i < len_x + 2; i++)
	{
		for (int j = 0; j < len_y + 2; j++)
			f1 << u[get_idx(i, j)] << "\t";
		f1 << endl;
	}
	f1.close();
	fname.str("");
	fname << "./out/"
		  << "id_" << id << "_v"
		  << "_" << out << ".dat";
	f1.open(fname.str().c_str(), ios_base::out);
	for (int i = 0; i < len_x + 2; i++)
	{
		for (int j = 0; j < len_y + 2; j++)
			f1 << v[get_idx(i, j)] << "\t";
		f1 << endl;
	}
	f1.close();
}
// setup the variables
void Simulator::setup(void)
{
	// calculating index
	tag_num = 0;
	my_idx_y = id / dim_x;
	my_idx_x = id - my_idx_y * dim_x;
	// calculating the size of each domain without boundary
	len_y = ceil((float)(Ny - 2) / dim_y);
	len_x = ceil((float)(Nx - 2) / dim_x);
	// starting index of the domain without boundary
	idx_x = my_idx_x * len_x;
	idx_y = my_idx_y * len_y;
	// finding the neighbors to do communication with
	neighbors[0] = id - 1;
	neighbors[1] = id - dim_x;
	neighbors[2] = id + 1;
	neighbors[3] = id + dim_x;
	if (my_idx_y == 0)
	{
		neighbors[1] = -1;
	}
	if (my_idx_x == 0)
	{
		neighbors[0] = -1;
	}
	// calculating the size of the boundary domain
	if (my_idx_y == dim_y - 1)
	{
		neighbors[3] = -1;
		len_y = Ny - 2 - ceil((float)(Ny - 2) / dim_y) * (dim_y - 1);
	}
	if (my_idx_x == dim_x - 1)
	{
		neighbors[2] = -1;
		len_x = Nx - 2 - ceil((float)(Nx - 2) / dim_x) * (dim_x - 1);
	}
	// cout << "process " << id << "have neighbors " << neighbors[0] << " " << neighbors[1] << " " << neighbors[2] << " " << neighbors[3] << endl;
	// calculating the subdomain size and allocating memory
	int total_size = (len_y + 2) * (len_x + 2);
	P = new double[total_size];
	P_old = new double[total_size];
	u = new double[total_size];
	u_old = new double[total_size];
	v = new double[total_size];
	v_old = new double[total_size];
	PPrhs = new double[total_size];
	// initializing the memory
	for (int i = 0; i < total_size; i++)
	{
		P[i] = 0;
		P_old[i] = 0;
		u[i] = 0;
		u_old[i] = 0;
		v[i] = 0;
		v_old[i] = 0;
		PPrhs[i] = 0;
	}
	// setting boundary condition
	if (my_idx_x == 0)
	{
		int idx = 0;
		for (int j = 0; j < len_y + 2; j++)
		{
			idx = j * (len_x + 2);
			P[idx] = P_max;
			P_old[idx] = P_max;
		}
	}
	// calculating dx dy
	dx = Lx / (Nx - 1);
	dy = Ly / (Ny - 1);

	t = 0.0;
	// build communication types
	build_col_type();
	build_row_type();
	// technically we only need 16 request
	req_list = new MPI_Request[32];

	// write configuration
#ifndef DO_TIMING
	if (id == 0)
		write_config();
#endif
}
// calculating laplace function
void Simulator::calculate_ppm_RHS_central(void)
{
	for (int i = 2; i < len_x; i++)
		for (int j = 2; j < len_y; j++)
		{
			PPrhs[get_idx(i, j)] = rho / dt * ((u[get_idx(i + 1, j)] - u[get_idx(i - 1, j)]) / (2. * dx) + (v[get_idx(i, j + 1)] - v[get_idx(i, j - 1)]) / (2. * dy));
		}
	// wait for the update and then update the boundary strip
	MPI_Waitall(req_cnt, req_list, MPI_STATUS_IGNORE);
	req_cnt = 0;

	for (int j = 1; j < len_y + 1; j += len_y - 1)
		for (int i = 1; i < len_x + 1; i++)
		{
			PPrhs[get_idx(i, j)] = rho / dt * ((u[get_idx(i + 1, j)] - u[get_idx(i - 1, j)]) / (2. * dx) + (v[get_idx(i, j + 1)] - v[get_idx(i, j - 1)]) / (2. * dy));
		}
	for (int i = 1; i < len_x + 1; i += len_x - 1)
		for (int j = 1; j < len_y + 1; j++)
		{
			PPrhs[get_idx(i, j)] = rho / dt * ((u[get_idx(i + 1, j)] - u[get_idx(i - 1, j)]) / (2. * dx) + (v[get_idx(i, j + 1)] - v[get_idx(i, j - 1)]) / (2. * dy));
		}
}
// set pressure boundary condition
void Simulator::set_pressure_BCs(void)
{
	if (my_idx_y == 0)
	{
		for (int i = 0; i < len_x + 2; i++)
		{
			P[get_idx(i, 0)] = P[get_idx(i, 1)];
		}
	}
	if (my_idx_y == dim_y - 1)
	{
		for (int i = 0; i < len_x + 2; i++)
		{
			P[get_idx(i, len_y + 1)] = P[get_idx(i, len_y)];
		}
	}
	if (my_idx_x == dim_x - 1)
	{
		for (int j = 0; j < len_y + 2; j++)
		{
			if (j + idx_y >= Ny / 2)
				P[get_idx(len_x + 1, j)] = P[get_idx(len_x, j)];
		}
	}
}
// calculating pressure by jacobi iteration
int Simulator::pressure_poisson_jacobi(double rtol = 1.e-5)
{
	double tol = 10. * rtol;
	int it = 0;
	// MPI_Request *req_list = new MPI_Request[16];

	while (tol > rtol)
	{
		swap(P, P_old);
		double sum_val = 0.0;
		tol = 0.0;
		it++;
		// Jacobi iteration
		for (int i = 1; i < len_x + 1; i++)
			for (int j = 1; j < len_y + 1; j++)
			{
				P[get_idx(i, j)] = 1.0 / (2.0 + 2.0 * (dx * dx) / (dy * dy)) * (P_old[get_idx(i + 1, j)] + P_old[get_idx(i - 1, j)] + (P_old[get_idx(i, j + 1)] + P_old[get_idx(i, j - 1)]) * (dx * dx) / (dy * dy) - (dx * dx) * PPrhs[get_idx(i, j)]);

				sum_val += fabs(P[get_idx(i, j)]);
				tol += fabs(P[get_idx(i, j)] - P_old[get_idx(i, j)]);
			}
		// send P to neighbors
		req_cnt = 0;
		sendhelper(P, req_cnt);
		// reduce sum_val and tol
		MPI_Allreduce(MPI_IN_PLACE, &sum_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &tol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		// wait for communications
		MPI_Waitall(req_cnt, req_list, MPI_STATUS_IGNORE);

		// set boundary conditions
		set_pressure_BCs();

		tol = tol / max(1.e-10, sum_val);
	}

	return it;
}

void Simulator::calculate_intermediate_velocity(int time_it)
{
	for (int i = 2; i < len_x; i++)
		for (int j = 2; j < len_y; j++)
		{
			// viscous diffusion
			u[get_idx(i, j)] = u_old[get_idx(i, j)] + dt * nu * ((u_old[get_idx(i + 1, j)] + u_old[get_idx(i - 1, j)] - 2.0 * u_old[get_idx(i, j)]) / (dx * dx) + (u_old[get_idx(i, j + 1)] + u_old[get_idx(i, j - 1)] - 2.0 * u_old[get_idx(i, j)]) / (dy * dy));
			v[get_idx(i, j)] = v_old[get_idx(i, j)] + dt * nu * ((v_old[get_idx(i + 1, j)] + v_old[get_idx(i - 1, j)] - 2.0 * v_old[get_idx(i, j)]) / (dx * dx) + (v_old[get_idx(i, j + 1)] + v_old[get_idx(i, j - 1)] - 2.0 * v_old[get_idx(i, j)]) / (dy * dy));
			// advection - upwinding
			if (u[get_idx(i, j)] > 0.0)
			{
				u[get_idx(i, j)] -= dt * u_old[get_idx(i, j)] * (u_old[get_idx(i, j)] - u_old[get_idx(i - 1, j)]) / dx;
				v[get_idx(i, j)] -= dt * u_old[get_idx(i, j)] * (v_old[get_idx(i, j)] - v_old[get_idx(i - 1, j)]) / dx;
			}
			else
			{
				u[get_idx(i, j)] -= dt * u_old[get_idx(i, j)] * (u_old[get_idx(i + 1, j)] - u_old[get_idx(i, j)]) / dx;
				v[get_idx(i, j)] -= dt * u_old[get_idx(i, j)] * (v_old[get_idx(i + 1, j)] - v_old[get_idx(i, j)]) / dx;
			}
			if (v[get_idx(i, j)] > 0.0)
			{
				u[get_idx(i, j)] -= dt * v_old[get_idx(i, j)] * (u_old[get_idx(i, j)] - u_old[get_idx(i, j - 1)]) / dy;
				v[get_idx(i, j)] -= dt * v_old[get_idx(i, j)] * (v_old[get_idx(i, j)] - v_old[get_idx(i, j - 1)]) / dy;
			}
			else
			{
				u[get_idx(i, j)] -= dt * v_old[get_idx(i, j)] * (u_old[get_idx(i, j + 1)] - u_old[get_idx(i, j)]) / dy;
				v[get_idx(i, j)] -= dt * v_old[get_idx(i, j)] * (v_old[get_idx(i, j + 1)] - v_old[get_idx(i, j)]) / dy;
			}
		}
	// if time_it == 1 it means this is the first run and it doesn't need to wait for the update from last run.
	// wait for the update and then update the boundary strip if this is not the first run
	// notice that here we actually are updating v_old and u_old instead of u and v becuase of the swap operation in solve_NS
	if (time_it != 1)
	{
		MPI_Waitall(req_cnt, req_list, MPI_STATUS_IGNORE);
		req_cnt = 0;
	}

	for (int j = 1; j < len_y + 1; j += len_y - 1)
		for (int i = 1; i < len_x + 1; i++)
		{
			// viscous diffusion
			u[get_idx(i, j)] = u_old[get_idx(i, j)] + dt * nu * ((u_old[get_idx(i + 1, j)] + u_old[get_idx(i - 1, j)] - 2.0 * u_old[get_idx(i, j)]) / (dx * dx) + (u_old[get_idx(i, j + 1)] + u_old[get_idx(i, j - 1)] - 2.0 * u_old[get_idx(i, j)]) / (dy * dy));
			v[get_idx(i, j)] = v_old[get_idx(i, j)] + dt * nu * ((v_old[get_idx(i + 1, j)] + v_old[get_idx(i - 1, j)] - 2.0 * v_old[get_idx(i, j)]) / (dx * dx) + (v_old[get_idx(i, j + 1)] + v_old[get_idx(i, j - 1)] - 2.0 * v_old[get_idx(i, j)]) / (dy * dy));
			// advection - upwinding
			if (u[get_idx(i, j)] > 0.0)
			{
				u[get_idx(i, j)] -= dt * u_old[get_idx(i, j)] * (u_old[get_idx(i, j)] - u_old[get_idx(i - 1, j)]) / dx;
				v[get_idx(i, j)] -= dt * u_old[get_idx(i, j)] * (v_old[get_idx(i, j)] - v_old[get_idx(i - 1, j)]) / dx;
			}
			else
			{
				u[get_idx(i, j)] -= dt * u_old[get_idx(i, j)] * (u_old[get_idx(i + 1, j)] - u_old[get_idx(i, j)]) / dx;
				v[get_idx(i, j)] -= dt * u_old[get_idx(i, j)] * (v_old[get_idx(i + 1, j)] - v_old[get_idx(i, j)]) / dx;
			}
			if (v[get_idx(i, j)] > 0.0)
			{
				u[get_idx(i, j)] -= dt * v_old[get_idx(i, j)] * (u_old[get_idx(i, j)] - u_old[get_idx(i, j - 1)]) / dy;
				v[get_idx(i, j)] -= dt * v_old[get_idx(i, j)] * (v_old[get_idx(i, j)] - v_old[get_idx(i, j - 1)]) / dy;
			}
			else
			{
				u[get_idx(i, j)] -= dt * v_old[get_idx(i, j)] * (u_old[get_idx(i, j + 1)] - u_old[get_idx(i, j)]) / dy;
				v[get_idx(i, j)] -= dt * v_old[get_idx(i, j)] * (v_old[get_idx(i, j + 1)] - v_old[get_idx(i, j)]) / dy;
			}
		}
	for (int i = 1; i < len_x + 1; i += len_x - 1)
		for (int j = 1; j < len_y + 1; j++)
		{
			// viscous diffusion
			u[get_idx(i, j)] = u_old[get_idx(i, j)] + dt * nu * ((u_old[get_idx(i + 1, j)] + u_old[get_idx(i - 1, j)] - 2.0 * u_old[get_idx(i, j)]) / (dx * dx) + (u_old[get_idx(i, j + 1)] + u_old[get_idx(i, j - 1)] - 2.0 * u_old[get_idx(i, j)]) / (dy * dy));
			v[get_idx(i, j)] = v_old[get_idx(i, j)] + dt * nu * ((v_old[get_idx(i + 1, j)] + v_old[get_idx(i - 1, j)] - 2.0 * v_old[get_idx(i, j)]) / (dx * dx) + (v_old[get_idx(i, j + 1)] + v_old[get_idx(i, j - 1)] - 2.0 * v_old[get_idx(i, j)]) / (dy * dy));
			// advection - upwinding
			if (u[get_idx(i, j)] > 0.0)
			{
				u[get_idx(i, j)] -= dt * u_old[get_idx(i, j)] * (u_old[get_idx(i, j)] - u_old[get_idx(i - 1, j)]) / dx;
				v[get_idx(i, j)] -= dt * u_old[get_idx(i, j)] * (v_old[get_idx(i, j)] - v_old[get_idx(i - 1, j)]) / dx;
			}
			else
			{
				u[get_idx(i, j)] -= dt * u_old[get_idx(i, j)] * (u_old[get_idx(i + 1, j)] - u_old[get_idx(i, j)]) / dx;
				v[get_idx(i, j)] -= dt * u_old[get_idx(i, j)] * (v_old[get_idx(i + 1, j)] - v_old[get_idx(i, j)]) / dx;
			}
			if (v[get_idx(i, j)] > 0.0)
			{
				u[get_idx(i, j)] -= dt * v_old[get_idx(i, j)] * (u_old[get_idx(i, j)] - u_old[get_idx(i, j - 1)]) / dy;
				v[get_idx(i, j)] -= dt * v_old[get_idx(i, j)] * (v_old[get_idx(i, j)] - v_old[get_idx(i, j - 1)]) / dy;
			}
			else
			{
				u[get_idx(i, j)] -= dt * v_old[get_idx(i, j)] * (u_old[get_idx(i, j + 1)] - u_old[get_idx(i, j)]) / dy;
				v[get_idx(i, j)] -= dt * v_old[get_idx(i, j)] * (v_old[get_idx(i, j + 1)] - v_old[get_idx(i, j)]) / dy;
			}
		}
}
// set velocity boundary conditions
void Simulator::set_velocity_BCs(void)
{
	if (my_idx_x == 0)
	{
		for (int j = 0; j < len_y + 2; j++)
		{
			u[get_idx(0, j)] = u[get_idx(1, j)];
		}
	}
	if (my_idx_x == dim_x - 1)
	{
		for (int j = 0; j < len_y + 2; j++)
		{
			if (j + idx_y < Ny / 2)
			{
				u[get_idx(len_x + 1, j)] = u[get_idx(len_x, j)];
			}
		}
	}
}

double Simulator::project_velocity(void)
{
	double vmax = 0.0;
	for (int i = 2; i < len_x; i++)
		for (int j = 2; j < len_y; j++)
		{
			u[get_idx(i, j)] = u[get_idx(i, j)] - dt * (1. / rho) * (P[get_idx(i + 1, j)] - P[get_idx(i - 1, j)]) / (2. * dx);
			v[get_idx(i, j)] = v[get_idx(i, j)] - dt * (1. / rho) * (P[get_idx(i, j + 1)] - P[get_idx(i, j - 1)]) / (2. * dy);

			double vel = sqrt(u[get_idx(i, j)] * u[get_idx(i, j)] + v[get_idx(i, j)] * v[get_idx(i, j)]);

			vmax = max(vmax, vel);
		}
	// wait for the update and then update the boundary strip
	// notice here we need to touch each position excactly once because the new value depend on the old value at the same position
	MPI_Waitall(req_cnt, req_list, MPI_STATUS_IGNORE);
	req_cnt = 0;
	for (int j = 1; j < len_y + 1; j += len_y - 1)
	{
		for (int i = 1; i < len_x + 1; i++)
		{
			u[get_idx(i, j)] = u[get_idx(i, j)] - dt * (1. / rho) * (P[get_idx(i + 1, j)] - P[get_idx(i - 1, j)]) / (2. * dx);
			v[get_idx(i, j)] = v[get_idx(i, j)] - dt * (1. / rho) * (P[get_idx(i, j + 1)] - P[get_idx(i, j - 1)]) / (2. * dy);

			double vel = sqrt(u[get_idx(i, j)] * u[get_idx(i, j)] + v[get_idx(i, j)] * v[get_idx(i, j)]);

			vmax = max(vmax, vel);
		}
	}
	for (int i = 1; i < len_x + 1; i += len_x - 1)
	{
		for (int j = 2; j < len_y; j++)
		{
			u[get_idx(i, j)] = u[get_idx(i, j)] - dt * (1. / rho) * (P[get_idx(i + 1, j)] - P[get_idx(i - 1, j)]) / (2. * dx);
			v[get_idx(i, j)] = v[get_idx(i, j)] - dt * (1. / rho) * (P[get_idx(i, j + 1)] - P[get_idx(i, j - 1)]) / (2. * dy);

			double vel = sqrt(u[get_idx(i, j)] * u[get_idx(i, j)] + v[get_idx(i, j)] * v[get_idx(i, j)]);

			vmax = max(vmax, vel);
		}
	}

	set_velocity_BCs();

	return vmax;
}
// helper function for communications
void Simulator::sendhelper(double *tosend, int &cnt)
{
	// check if this process have left neighbour
	if (neighbors[0] != -1)
	{
		// cout<<id<<" sending 0 to " << neighbors[0] << endl;
		MPI_Isend(&tosend[get_idx(1, 1)], 1, Datatype_col, neighbors[0], tag_num, MPI_COMM_WORLD, &req_list[cnt]);
		cnt++;
		MPI_Irecv(&tosend[get_idx(0, 1)], 1, Datatype_col, neighbors[0], tag_num, MPI_COMM_WORLD, &req_list[cnt]);
		cnt++;
	}
	// check if this process have up neighbour
	if (neighbors[1] != -1)
	{
		// cout<<id<<" sending 1 to " << neighbors[1] << endl;

		MPI_Isend(&tosend[get_idx(1, 1)], 1, Datatype_row, neighbors[1], tag_num, MPI_COMM_WORLD, &req_list[cnt]);
		cnt++;
		MPI_Irecv(&tosend[get_idx(1, 0)], 1, Datatype_row, neighbors[1], tag_num, MPI_COMM_WORLD, &req_list[cnt]);
		cnt++;
	}
	// check if this process have right neighbour
	if (neighbors[2] != -1)
	{
		// cout<<id<<" sending 2 to " << neighbors[2] << endl;

		MPI_Isend(&tosend[get_idx(len_x, 1)], 1, Datatype_col, neighbors[2], tag_num, MPI_COMM_WORLD, &req_list[cnt]);
		cnt++;
		MPI_Irecv(&tosend[get_idx(len_x + 1, 1)], 1, Datatype_col, neighbors[2], tag_num, MPI_COMM_WORLD, &req_list[cnt]);
		cnt++;
	}
	// check if this process have bottom neighbour
	if (neighbors[3] != -1)
	{
		// cout<<id<<" sending 3 to " << neighbors[3] << endl;

		MPI_Isend(&tosend[get_idx(1, len_y)], 1, Datatype_row, neighbors[3], tag_num, MPI_COMM_WORLD, &req_list[cnt]);
		cnt++;
		MPI_Irecv(&tosend[get_idx(1, len_y + 1)], 1, Datatype_row, neighbors[3], tag_num, MPI_COMM_WORLD, &req_list[cnt]);
		cnt++;
	}
	tag_num++;
}
// solve NS functions
void Simulator::solve_NS(void)
{
	double vel_max = 0.0;
	int time_it = 0;
	int its;
	int out_it = 0;
	double t_out = dt_out;
	// write inital values to disk
#ifndef EXCLUDE_FILE_WRITE
	grids_to_file(out_it);
#endif
	// iterate through time
	while (t < t_end)
	{
		if (vel_max > 0.0)
		{
			dt = min(courant * min(dx, dy) / vel_max, dt_min);
		}
		else
			dt = dt_min;
		// MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
		// if(id==0)
		// 	cout << "dt" << dt << endl;
		t += dt;
		time_it++;
		swap(u, u_old);
		swap(v, v_old);

		calculate_intermediate_velocity(time_it);
		req_cnt = 0;
		// send u v
		sendhelper(u, req_cnt);
		sendhelper(v, req_cnt);
		// MPI_Waitall(req_cnt, req_list, MPI_STATUS_IGNORE);
		// req_cnt = 0;
		calculate_ppm_RHS_central();
		its = pressure_poisson_jacobi(1.e-5);
		// send P
		sendhelper(P, req_cnt);
		// MPI_Waitall(req_cnt, req_list, MPI_STATUS_IGNORE);
		// req_cnt = 0;
		vel_max = project_velocity();
		// send u v
		sendhelper(u, req_cnt);
		sendhelper(v, req_cnt);

		// MPI_Waitall(req_cnt, req_list, MPI_STATUS_IGNORE);
		// req_cnt = 0;

		// reduce max velocity, this can be done before we send the data since sending the data
		// doesn't create new data
		MPI_Allreduce(MPI_IN_PLACE, &vel_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		if (t >= t_out)
		{
			// print infomation and save to disk
			out_it++;
			t_out += dt_out;
#ifndef DO_TIMING
			if (id == 0)
			{
				cout << time_it << ": " << t << " Jacobi iterations: " << its << " vel_max: " << vel_max << endl;
			}
#endif
#ifndef EXCLUDE_FILE_WRITE
			grids_to_file(out_it);
#endif
		}
		if (t >= t_end)
		{
			MPI_Waitall(req_cnt, req_list, MPI_STATUS_IGNORE);
			req_cnt = 0;
		}
	}
}

int main(int argc, char *argv[])
{
#ifdef DO_TIMING
	auto start = chrono::high_resolution_clock::now();
#endif

	MPI_Init(&argc, &argv);
	Simulator sim;
	// read user input from file if given
	if (argc == 2)
	{
		sim.read_config_file(argv[1]);
	}
	sim.setup();
#ifdef DO_TIMING
	MPI_Barrier(MPI_COMM_WORLD);
	auto finish_setup = chrono::high_resolution_clock::now();
	if (sim.id == 0)
	{
		std::chrono::duration<double> elapsed = finish_setup - start;
		cout << setprecision(5);
		cout << "The code took " << elapsed.count() << "s to initialize" << endl;
	}
#endif
	sim.solve_NS();
#ifdef DO_TIMING
	MPI_Barrier(MPI_COMM_WORLD);
	auto finish = chrono::high_resolution_clock::now();
	if (sim.id == 0)
	{
		std::chrono::duration<double> elapsed = finish - start;
		cout << setprecision(5);
		cout << "The code took " << elapsed.count() << "s to finish" << endl;
	}
#endif
	return 0;
}
