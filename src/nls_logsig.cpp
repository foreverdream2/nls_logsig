#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlin.h>

using namespace std;

#define check(flag,msg) if(flag){cerr << msg << endl;exit(1);}
#define SQR(x) ((x)*(x))
#define npar 3	// 3 parameters model

// fit type of results: Levenberg Marquardt or Grid Search
#define FIT_SUCCESS 0
#define FIT_MAXITER 1
#define FIT_OUTBOX 2


typedef pair<double, double> range_pair;

// global parameters
double Ainf_upper = 1, Ainf_lower = 0,
		Hill_upper = 10, Hill_lower = 0,
		EC50_upper = 10, EC50_lower = -10,
		grid_step = 1e-2, tolerance = 1e-8, tol_upper = 20;

size_t max_iter = 1e3;

// public workspace
double *workspace = NULL;

const gsl_multifit_fdfsolver_type *T = gsl_multifit_fdfsolver_lmsder;

gsl_matrix *J;
gsl_vector *para;
gsl_multifit_function_fdf func;

struct drug_curve
{
  double *dose, *activity;
  size_t N;
};


int logsig(const gsl_vector *para, void *data, gsl_vector *result)
{
	drug_curve *ptr = (struct drug_curve *)data;

	double t,
		Ainf = gsl_vector_get (para, 0),
		EC50 = gsl_vector_get (para, 1),
		Hill = gsl_vector_get (para, 2);

	for (size_t i = 0; i < ptr->N; i++)
	{
		t = 1 + exp(Hill*(ptr->dose[i]-EC50));
		t = Ainf * (1-1/t);
		gsl_vector_set (result, i, t - ptr->activity[i]);
	}

	return GSL_SUCCESS;
}

int logsig_jac(const gsl_vector *para, void *data, gsl_matrix * J)
{
	drug_curve *ptr = (struct drug_curve *)data;

	double t, delta, c, x,
		Ainf = gsl_vector_get (para, 0),
		EC50 = gsl_vector_get (para, 1),
		Hill = gsl_vector_get (para, 2);

	for (size_t i = 0; i < ptr->N; i++)
	{
		x = ptr->dose[i];
		t = exp(Hill*(x-EC50));
		delta = 1/(1 + t);
		c = Ainf * SQR(delta) *t;

		gsl_matrix_set (J, i, 0, 1-delta);		// Ainf
		gsl_matrix_set (J, i, 1, -c*Hill);		// EC50
		gsl_matrix_set (J, i, 2, c*(x-EC50));	// Hill
    }

	return GSL_SUCCESS;
}


double AUC(const double Ainf, const double Hill, const double EC50, const double upper, const double lower)
{
  if(Hill == 0)
    return Ainf/2;
  else{
	  double t_upper = Hill*(upper-EC50), t_lower = Hill*(lower-EC50);

	  // numerical precision control
	  if(t_upper < tol_upper) t_upper = log(exp(t_upper) +1);
	  if(t_lower < tol_upper) t_lower = log(exp(t_lower) +1);

	  return Ainf/Hill*(t_upper-t_lower)/(upper-lower);
  }
}



void resize_matrix(gsl_matrix *m, const size_t size1, const size_t size2)
{
	m->size1 = size1;
	m->tda = m->size2 = size2;
	m->block->size = m->size1*m->size2;
}


double nls_lm(
	const double dose[], const double activity[], const size_t N,
	double &Ainf, double &EC50, double &Hill)
{
	int status, info;

	double MSE = numeric_limits<double>::max();

	// resize space
	drug_curve *ptr = (drug_curve*)(func.params);
	ptr->N = func.n = N;
	resize_matrix(J, N, npar);

	// initialized new solver
	gsl_multifit_fdfsolver *solver = gsl_multifit_fdfsolver_alloc (T, N, npar);

	// initialize
	gsl_vector_set (para, 0, 1),	 // Ainf
	gsl_vector_set (para, 1, 0), // EC50
	gsl_vector_set (para, 2, 1); // Hill

	gsl_multifit_fdfsolver_set(solver, &func, para);

	status = gsl_multifit_fdfsolver_driver(solver, max_iter, tolerance, tolerance, 0, &info);

	if(status == GSL_SUCCESS)
	{
		gsl_blas_ddot(solver->f, solver->f, &MSE);
		MSE /= N;
	}
	//else MSE is max double

	Ainf = gsl_vector_get(solver->x, 0);
	EC50 = gsl_vector_get(solver->x, 1);
	Hill = gsl_vector_get(solver->x, 2);

	gsl_multifit_fdfsolver_free(solver);

	return MSE;
}


double grid_search(
	const double dose[], const double activity[], const size_t N,
	double &Ainf, double &EC50, double &Hill)
{
	size_t i;
	double RSS = numeric_limits<double>::max(), A, E, H, res;

	for(E=EC50_lower; E<=EC50_upper; E+=grid_step)
	{
		for(H=Hill_lower; H<= Hill_upper; H+=grid_step)
		{
			if (fabs(H) < tolerance) continue;

			// calculate delta array
			for(i=0;i<N;i++) workspace[i] = 1-1/(1+exp(H*(dose[i]-E)));

			// calculate A
			for(res=A=0,i=0;i<N;i++)
			{
				A += activity[i] * workspace[i];
				res += SQR(workspace[i]);
			}

			A /= res;
			A = min<double>(max<double>(A, Ainf_lower), Ainf_upper);

			for(res=0,i=0;i<N;i++) res += SQR(activity[i]-A*workspace[i]);

			if(res < RSS)
			{
				RSS = res;
				Ainf = A;
				EC50 = E;
				Hill = H;
			}
		}
	}

	return RSS/N;
}


double fit_parameters(
	const double dose[], const double activity[], const size_t N,
	double &Ainf, double &EC50, double &Hill, size_t &fit_type)
{
	double MSE = nls_lm(dose, activity, N, Ainf, EC50, Hill);

	// decide fit type first
	if(MSE == numeric_limits<double>::max())
		fit_type = FIT_MAXITER;
	else{
		bool boxflag =
			Ainf_lower <= Ainf && Ainf <= Ainf_upper &&
			Hill_lower < Hill && Hill <= Hill_upper &&
			EC50_lower <= EC50 && EC50 <= EC50_upper;

		if(boxflag)
			fit_type = FIT_SUCCESS;
		else
			fit_type = FIT_OUTBOX;
	}

	if(fit_type != FIT_SUCCESS) MSE = grid_search(dose, activity, N, Ainf, EC50, Hill);

	return MSE;
}



double convert_to_double(const string n)
{
	if(n == "NA")
		return numeric_limits<double>::quiet_NaN();
	else{
		istringstream iss(n);
		double v;

		iss >> v;
		check(iss.fail(), "Cannot convert double for " << n)

		return v;
	}
}


void split(const string &line, vector<double> &vec, const bool logflag = false)
{
	double v;
	string n;
	istringstream iss(line);
	for(vec.clear(); getline(iss,n,','); vec.push_back(v))
	{
		v = convert_to_double(n);
		if(logflag) v = log(v);
	}
}


template <class T>
void load_bound(const string &line, T &lower, T &upper)
{
	string upper_s, lower_s;
	istringstream iss(line);

	getline(iss, lower_s, ',');
	check(iss.eof(), "Cannot find bound pairs for " << line << ". Please use format \"lower,upper\"")

	getline(iss, upper_s, ',');
	iss.clear();

	iss.str(lower_s);
	iss >> lower;
	check(iss.fail(), "Cannot parse lower bound " << lower_s)
	iss.clear();

	iss.str(upper_s);
	iss >> upper;
	check(iss.fail(), "Cannot parse upper bound " << upper_s)
	iss.clear();

	check(upper <= lower, "upper bound " << upper << " <= lower bound " << lower)
}


int main(int argc, char *argv[])
{
	size_t i, N, total, fit_type, parseCnt = (argc-1)/2, progress_inx, progress_step,

			// running range among all lines, used for parallel processing
			run_upper = string::npos, run_lower = string::npos;

	double EC50=0, Hill=0, Ainf=0, MSE=0, upperv=0, lowerv=0;

	bool log_dose = true;

	ifstream fin;
	istringstream iss;
	string type, line, input, output, drug, cell;

	// for parsing number vectors
	vector<double> vec;
	vector<double>::iterator viter;

	// drug -> dose range map
	map<string, range_pair> range_map;
	map<string, range_pair>::iterator miter;

	// data array for each line
	double *dose, *activity;

	// wrapper of data array for NLS solver
	drug_curve *ptr = new drug_curve;

	// hints
	if ( argc < 5 )
	{
		if(argc == 2 && string(argv[1]) == "-help")
		{
			cout << endl;
			cout << "Non-linear least square for log-sigmoid drug response\n" << endl;

			cout << "Usage: nls_logsig -i drug_curve -o output\n"<<endl;
			cout << "Parameter [lower, upper] bound:" << endl;
			cout << "\t-a Ainf. Default: " << Ainf_lower << ',' << Ainf_upper << endl;
			cout << "\t-h Hill. Default: " << Hill_lower << ',' << Hill_upper << endl;
			cout << "\t-e EC50. Default: " << EC50_lower << ',' << EC50_upper << endl;
			cout << endl;

			cout << "Other options:" << endl;
			cout << "\t-l log transform dose (0 or 1). Default: " << log_dose << endl;
			cout << "\t-s step size in grid search. Default: " << grid_step << endl;
			cout << "\t-t tolerance. Default: " << tolerance << endl;
			cout << "\t-n max iteration. Default: " << max_iter << endl;
			cout << "\t-r run range of [lower,upper). Default: empty" << endl;
			cout << endl;

			cout << "Report bugs to Peng Jiang (peng.jiang.software@gmail.com)\n" <<endl;
			exit(0);
		}else{
			cerr << "Insufficient number of arguments, do \"nls_logsig -help\" for help."<<endl;
			exit(1);
		}
	}

	// read in all parameters
	for(i=0;i<parseCnt;i++)
	{
		type = argv[2*i+1];
		line = argv[2*i+2];

		if(type == "-i"){
			input = line;

		}else if (type == "-o"){
			output = line;

		}else if (type == "-a"){
			load_bound(line, Ainf_lower, Ainf_upper);

		}else if (type == "-h"){
			load_bound(line, Hill_lower, Hill_upper);

		}else if (type == "-e"){
			load_bound(line, EC50_lower, EC50_upper);

		}else if(type == "-r"){
			load_bound(line, run_lower, run_upper);

		}else if(type == "-s"){
			grid_step = convert_to_double(line);
			check(grid_step <=0, "grid step " << grid_step << " <=0")

		}else if(type == "-t"){
			tolerance = convert_to_double(line);
			check(tolerance <=0, "tolerance " << tolerance << " <=0")

		}else if(type == "-l"){
			log_dose = (line == "1");
			check(line != "1" && line != "0", "Please input 1 or 0. You input " << line)

		}else if(type == "-n"){
			max_iter = atoi(line.c_str());

		}else if (type == "-help"){
			cerr << "Please don't use \"-help\" as parameter input." << endl;
			exit(1);

		}else{
			cerr << "Cannot recognize parameter \""<< type << "\"." << endl;
			exit(1);
		}
	}

	check(input.empty(), "Cannot find drug curve input")
	check(output.empty(), "Cannot find output name")

	// pass 1, get overall number of lines and maximum data length
	fin.open(input.c_str());
	check(fin.fail(), "Cannot open file " << input);

	for(N=total=0; getline(fin, line, '\n'); total++)
	{
		iss.str(line);
		getline(iss, cell, '\t');
		getline(iss, drug, '\t');
		getline(iss, line, '\t');
		iss.clear();

		// analyze dose for drug specific range
		split(line, vec, log_dose);

		upperv = *max_element(vec.begin(), vec.end());
		lowerv = *min_element(vec.begin(), vec.end());

		miter = range_map.find(drug);

		if(miter == range_map.end()){
			range_map[drug] = range_pair(upperv, lowerv);
		}else{
			miter->second.first = max<double>(miter->second.first, upperv);
			miter->second.second = min<double>(miter->second.second, lowerv);
		}

		N = max<size_t>(N, vec.size());
	}

	fin.close();
	fin.clear();

	// allocate space
	dose = new double[N];
	activity = new double[N];
	workspace = new double[N];

	ptr->dose = dose;
	ptr->activity = activity;

	para = gsl_vector_alloc(npar);
	J = gsl_matrix_alloc(N, npar);

	// hook up functions for Levenberg-Marquardt
	func.f = &logsig;
	func.df = &logsig_jac;
	func.p = npar;
	func.params = (void*)ptr;

	// upper bound of tolerence
	tol_upper = -log(tolerance);

	// prepare progress report
	progress_step = max<size_t>(total/100, 1);

	ofstream fout(output.c_str());
	fout << "Cell\tDrug\tAUC\tMSE\tAinf\tEC50\tHill\tFit\n";

	// pass 2, run fittings
	fin.open(input.c_str());

	for(progress_inx=0; getline(fin, line, '\n'); progress_inx++)
	{
		// run range control, used for parallel processing
		if(run_lower != string::npos && progress_inx < run_lower) continue;
		if(run_upper != string::npos && progress_inx >= run_upper) continue;

		// progress report
		if(progress_inx % progress_step == 0)
			cout << round(100.0*progress_inx/total) << '%' << endl;

		// read lines
		iss.str(line);
		getline(iss, cell, '\t');
		getline(iss, drug, '\t');

		// analyze dose with log transformation
		getline(iss, line, '\t');
		split(line, vec, log_dose);

		N = vec.size();
		for(i=0;i<N;i++) dose[i] = vec[i];

		// analyze response
		getline(iss, line, '\t');
		split(line, vec);
		check(vec.size()!=N, "dose and activity length mismatch on " << cell << '\t' << drug)
		for(i=0;i<N;i++) activity[i] = vec[i];

		iss.clear();

		// drug dose range for integration
		miter = range_map.find(drug);
		check(miter == range_map.end(), "impossible missing drugs at second round")

		upperv = miter->second.first;
		lowerv = miter->second.second;

		// fit parameters
		MSE = fit_parameters(dose, activity, N, Ainf, EC50, Hill, fit_type);

		fout << cell << '\t' << drug << '\t' << AUC(Ainf, Hill, EC50, upperv, lowerv) << '\t'
			<< MSE << '\t' << Ainf << '\t'  << EC50 << '\t' <<  Hill << '\t'
			<< fit_type << '\n';
	}

	fout.close();
	fin.close();

	delete[] dose;
	delete[] activity;
	delete[] workspace;
	delete ptr;

	resize_matrix(J, N, npar);
	gsl_matrix_free(J);
	gsl_vector_free(para);

	return 0;
}
