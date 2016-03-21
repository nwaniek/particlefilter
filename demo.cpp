#include "particlefilter.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>

/*
 * some static stuff to keep the demo simple
 */

// the agent lives in a suqare world
static double world_size = 150.0;

// random number generator
static std::mt19937_64 rng(1337);
static std::uniform_real_distribution<double> unif{0.0, 1.0};


struct Point {
	double x, y;
};

/* the world has just four landmarks placed randomly */
std::vector<Point> landmarks{
	{15.0, 12.0},
	{85.0, 74.0},
	{ 7.0, 44.0},
	{65.0, 65.0},
	{123.0, 77.0}};

/*
 * Representation of a measurement done by an agent. Each measurement is a
 * distance towards a landmark
 */
struct Measurement {
	std::vector<double> dist;
};



/**
 * Agent that has a location x, y, and an orientation. It does some random walk
 * through the square world
 */
struct Agent
{
	double x = world_size / 2.0;
	double y = world_size / 2.0;
	double o = M_PI;
	double speed = 5.0;
	double sensor_noise = 5.0;

	/*
	 * agent moves randomly around in the world and takes really noisy
	 * measurements
	 */
	std::normal_distribution<double> nd_speed{0.0, 0.4};
	std::normal_distribution<double> nd_rotate{0.0, 0.3};
	std::normal_distribution<double> nd_sensor{0.0, sensor_noise};

	void move() {
		double tmp_o = o;
		double tmp_x = x;
		double tmp_y = y;

		// probe for a valid movement
		while (true) {

			tmp_o = tmp_o + nd_rotate(rng);
			while (tmp_o < 0) tmp_o += 2.0 * M_PI;
			while (tmp_o > (2.0*M_PI)) tmp_o -= 2.0 * M_PI;

			double dist = speed + nd_speed(rng);
			tmp_x = x + std::cos(tmp_o) * dist;
			tmp_y = y + std::sin(tmp_o) * dist;

			if (!(tmp_x < 0.0 || tmp_y < 0.0 || tmp_x > world_size || tmp_y > world_size)) break;
		}

		x = tmp_x;
		y = tmp_y;
		o = tmp_o;
	}

	Measurement sense() {
		Measurement Z;
		for (size_t i = 0; i < landmarks.size(); i++) {
			double xdist = x - landmarks[i].x;
			double ydist = y - landmarks[i].y;
			double dist = std::sqrt(xdist * xdist +  ydist * ydist);
			dist += nd_sensor(rng);
			Z.dist.push_back(dist);
		}
		return Z;
	}

};

std::ostream& operator<< (std::ostream &stream, const Agent &a)
{
	stream << "[x=" << a.x << ", y=" << a.y << ", o=" << a.o << "]";
	return stream;
}


template <typename RealType = double>
RealType gaussian (const RealType mu, const RealType sigma, const RealType x)
{
	auto var = sigma * sigma;
	auto dist = x - mu;
	dist *= dist;
	return (1. / std::sqrt(2. * M_PI * var)) * std::exp(-.5 * dist / var);
}


/**
 * Implementation of a state estimate particle for the agent. In principle, this
 * could be incorporated into Agent, but usually it's better pratice to keep the
 * state estimation separate from the logic implementing the Agent.
 */
template <typename MeasurementType, typename RealType>
struct particle :
	bayes::particle<MeasurementType, RealType>
{
	/*
	 * hijack the Agent class to reduce code we have to write
	 */
	Agent state;

	/*
	 * initialize the particle state to some random location
	 */
	particle() {
		state.x = unif(rng) * world_size;
		state.y = unif(rng) * world_size;
		state.o = unif(rng) * 2.0 * M_PI;
	};

	/*
	 * compute the importance weight. This applies the measurement noise
	 * model to the measurement and yields how likely the particle actually
	 * is.
	 */
	RealType compute_weight(const MeasurementType *Z) override
	{
		double prob = 1.0;

		for (size_t i = 0; i < landmarks.size(); i++) {
			double xdist = state.x - landmarks[i].x;
			double ydist = state.y - landmarks[i].y;
			double dist = std::sqrt(xdist*xdist + ydist*ydist);

			prob *= gaussian(dist, state.sensor_noise, Z->dist[i]);
		}

		return prob;
	}

	// apply the motion model to the particle
	void move() override
	{
		state.move();
	}
};


/*
 * evaluation of particle filter estimate
 */
double evaluate(size_t N, const Agent &a, std::vector<particle<Measurement, double>> ps)
{
	double err  = 0.0;
	for (size_t i = 0; i < N; i++) {
		double dx = ps[i].state.x - a.x;
		double dy = ps[i].state.y - a.y;
		err += std::sqrt(dx*dx + dy*dy);
	}
	return err / double(N);
}



int main()
{
	constexpr size_t Nparticles = 1000;
	Agent agent;
	bayes::particle_filter<particle<Measurement, double>, double> pf(Nparticles);

	// number of iterations to loop in this demo
	constexpr size_t Tmax = 100;

	size_t t = 0;
	while (t++ < Tmax) {
		agent.move();

		// get a measurement from the agent
		auto Z = agent.sense();

		std::cout << std::fixed << std::setprecision(3)
			<< agent << ", quality=" << evaluate(Nparticles, agent, pf.particles)
			<< "\n";

		pf.move();
		pf.compute_weights(&Z);
		pf.resample();
	}
}
