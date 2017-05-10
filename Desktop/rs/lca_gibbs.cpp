#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <Eigen/StdVector>
using namespace std;
using namespace Eigen;
#include <RcppEigen.h>
#include <Rcpp.h>
#include <boost/random/beta_distribution.hpp>
#define MAXBUFSIZE ((int) 1e6)

///////////////////////// Golbal Variables /////////////////
/////// random number generator /////////
random_device rd;
default_random_engine generator(rd());
//const double euler_const = 0.5772156649;
///////////////////////////////////////////////////////////

// MatrixXi readMatrix(const char *filename){
//     int cols = 0, rows = 0;
//     double buff[MAXBUFSIZE];
//     // Read numbers from file into buffer.
//     ifstream infile;
//     infile.open(filename);
//     while (! infile.eof())
//         {
//         string line;
//         getline(infile, line);
//         int temp_cols = 0;
//         stringstream stream(line);
//         while(! stream.eof())
//             stream >> buff[cols*rows+temp_cols++];
//         if (temp_cols == 0)
//             continue;
//         if (cols == 0)
//             cols = temp_cols;
//         rows++;
//         }
//     infile.close();
//     rows--;
//     // Populate matrix with numbers.
//     MatrixXi result(rows,cols);
//     for (int i = 0; i < rows; i++)
//         for (int j = 0; j < cols; j++)
//             result(i,j) = buff[ cols*i+j ];
//     return result;
// }

void removeColumn(MatrixXi& matrix, int colToRemove){
    int numRows = matrix.rows();
    int numCols = matrix.cols()-1;
    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);
    matrix.conservativeResize(numRows,numCols);
}

void removeCell(VectorXi& vec, int cellToRemove){
	int ind_max = vec.size()-1;
	if (cellToRemove < ind_max)
		vec.segment(cellToRemove,ind_max-cellToRemove) = vec.tail(ind_max-cellToRemove);
	vec.conservativeResize(ind_max);
}

// [[Rcpp::depends(RcppEigen)]]
using Eigen::Dense;

// [[Rcpp::export]]
Rcpp::List lca_gibbs(MatrixXi x, int n_itr, int init_dim, double init_alpha){
	//////////// variable declaration ////////
	MatrixXi z_history;
	double alpha,log_p,alpha_dir_sum,alpha_1,alpha_2,v,pi,pi_ratio,thres;
	int N,J,n_cat=2,k;
	VectorXi z, n_list;
	VectorXd init_prob,alpha_history,prob_vec,alpha_dir;
	vector<MatrixXi> c_list;
	/////////////////////////////////
	//x = readMatrix("data.txt");
	N = x.rows();
	J = x.cols();
	////////////////////////////////
	////////////// setting priors /////////
	alpha_dir.resize(n_cat);
	alpha_dir.fill(1.0);
	alpha = init_alpha;
	alpha_dir_sum = alpha_dir.sum();
	alpha_1 = 1;
	alpha_2 = 1;
	//////////////////////////////
	//////////// initialize the chain ////////
	z.resize(N);
	n_list.resize(init_dim);
	n_list.fill(0);
	c_list.resize(J);
	for (int i = 0; i < J; i++){
		c_list[i].resize(n_cat,init_dim);
		c_list[i].fill(0);
	}
	init_prob.resize(init_dim);
	init_prob.fill(1.0);
	discrete_distribution <int> rdis {init_prob.data(),init_prob.data()+init_prob.size()};
	for (int i = 0; i < z.size(); i++){
		z(i) = rdis(generator);
		n_list(z(i))++;
		for (int j = 0; j < J; j++)
			c_list[j](x(i,j),z(i))++;
	}
	////////////////////////////////
	///////// Collapsed Gibbs with CRP prior /////////////
	z_history.resize(n_itr,N);
	z_history.row(0) = z;
	alpha_history.resize(n_itr);
	alpha_history(0) = alpha;
	for (int itr = 1; itr < n_itr; itr++){
		for (int i = 0; i < N; i++){
			n_list(z(i))--;
			for (int j = 0; j < J; j++)
				c_list[j](x(i,j),z(i))--;
			if (n_list(z(i)) == 0){
				removeCell(n_list,z(i));
				for (int j = 0; j < J; j++)
					removeColumn(c_list[j],z(i));
				for (int j = 0; j < z.size(); j++){
					if (z(j) > z(i)) z(j)--;
				}
			}
			k = n_list.size();
			prob_vec.resize(k+1);
			for (int m = 0; m < k; m++){
				log_p = 0;
				for (int j = 0; j < J; j++)
					log_p += log(alpha_dir(x(i,j)) + c_list[j](x(i,j),m)) - log(alpha_dir_sum+n_list(m));
				prob_vec(m) = exp(log(n_list(m)) + log_p);
			}
			prob_vec(k) = exp(log(1.0/n_cat)*J + log(alpha));
			discrete_distribution <int> rdis {prob_vec.data(),prob_vec.data()+prob_vec.size()};
			z(i) = rdis(generator);
			if (z(i) == k){
				n_list.conservativeResize(k+1);
				n_list(k) = 1;
				for (int j = 0; j < J; j++){
					c_list[j].conservativeResize(n_cat,k+1);
					c_list[j].col(k).fill(0);
					c_list[j](x(i,j),k)++;
				}
			}
			else{
				n_list(z(i))++;
				for (int j = 0; j < J; j++)
					c_list[j](x(i,j),z(i))++;
			}
		}
		z_history.row(itr) = z;

		/////// update the dispersion parameter for CRP prior //////////
		// k = n_list.size();
		// for (int i = 0; i < alpha_prob.size(); i++){
		// 	candidate = (i+1) * 0.01;
		// 	alpha_prob(i) = k * log(candidate) + lgamma(candidate) - lgamma(candidate+N) + (alpha_1-1)*log(candidate)-candidate/alpha_2;
		// 	if (alpha_prob(i) < temp_min) temp_min = alpha_prob(i);
		// }
		// //temp_min.fill(alpha_prob.minCoeff());
		// //alpha_prob = alpha_prob - temp_min;
		// for (int i = 0; i < alpha_prob.size(); i++) alpha_prob(i) = exp(alpha_prob(i)-temp_min);
		// discrete_distribution <int> rdis {alpha_prob.data(),alpha_prob.data()+alpha_prob.size()};
		// alpha = (rdis(generator)+1) * 0.01;
		// alpha = 0.3;

		///////////////////// Update the dispersion parameter through augmentation /////////
		k = n_list.size();
		boost::random::beta_distribution<double> rbeta(alpha+1,N);
		v = rbeta(generator);
		pi_ratio = (alpha_1+k-1)  / (N*(alpha_2-log(v)));
		pi = pi_ratio / (1+pi_ratio);
		uniform_real_distribution<double> runif(0.0,1.0);
		thres = runif(generator);
		if (thres > pi){
			gamma_distribution<double> rgamma(alpha_1+k-1,1/(alpha_2-log(v)));
			alpha = rgamma(generator);
		}
		else{
			gamma_distribution<double> rgamma(alpha_1+k,1/(alpha_2-log(v)));
			alpha = rgamma(generator);
		}
		//alpha = 0.1;
		alpha_history(itr) = alpha;
		//v_history(itr) = v;
	}
	return Rcpp::List::create(Rcpp::Named("z_samples")=z_history,Rcpp::Named("alpha_samples")=alpha_history);
}

// [[Rcpp::export]]
MatrixXd get_item_param(MatrixXi x, VectorXi z, int n_cat, int n_dim){
	int N,J;
	MatrixXd beta;
	MatrixXd temp;
	
	N = x.rows();
	J = x.cols();

	beta.resize(n_cat*J,n_dim);
	temp.resize(n_cat,n_dim);
	for (int i = 0; i < J; i++){
		temp.fill(0);
		for (int j = 0; j < N; j++)
			temp(x(j,i),z(j))++;
		for (int j = 0; j < n_dim; j++)
			beta.col(j).segment(i*n_cat,n_cat) = temp.col(j) / temp.col(j).sum();
	}
	return beta;
}

// [[Rcpp::export]]
double get_rmse(MatrixXd est, MatrixXd gen, MatrixXi perm){
	int N,J,C;
	VectorXd temp;
	double sq_error, min_rmse;

	N = perm.rows();
	J = est.rows();
	C = est.cols();
	temp.resize(C);
	sq_error = 0;
	for (int i = 0; i < J; i++){
		sq_error += (est.row(i) - gen.row(i)).squaredNorm();
	}
	min_rmse = sqrt(sq_error/(J*C));
	for (int itr = 0; itr < N; itr++){
		sq_error = 0;
		for (int i = 0; i < J; i++){
			for (int j = 0; j < C; j ++)
				temp(j) = gen(i,perm(itr,j));
			sq_error += (est.row(i) - temp).squaredNorm();
		}
		if (sqrt(sq_error) < min_rmse) min_rmse = sqrt(sq_error/(J*C)); 	
	}
	return (min_rmse);
}