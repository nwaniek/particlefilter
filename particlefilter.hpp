#pragma once
#ifndef __PF_HPP__EA28733A_563C_4851_AC42_A55C2DE7A08C
#define __PF_HPP__EA28733A_563C_4851_AC42_A55C2DE7A08C

#include <cstddef>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>


namespace bayes {

template <typename MeasurementType, typename RealType = double>
struct particle {
	typedef particle<MeasurementType, RealType> ptype;
	typedef RealType rtype;
	typedef MeasurementType mtype;

	RealType weight;

	virtual RealType compute_weight(const MeasurementType *Z) = 0;
	virtual void move() = 0;
};


/*
 * the low variance resampler selects N indices out of a set of particles
 */
template <size_t N, typename ParticleType, typename RealType = double>
struct low_variance_resampler
{
	low_variance_resampler() {
		// seed the rng with a time dependent sequence
		uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::seed_seq seq{uint32_t(seed & 0xFFFFFFFF), uint32_t(seed >> 32)};
		rng.seed(seq);
	}

	explicit low_variance_resampler(int seed) {
		// seed the rng with a fixed value
		rng.seed(seed);
	}

	std::vector<size_t> resample(const std::vector<ParticleType> &particles) {
		size_t index = size_t(unif(rng) * N);
		std::vector<size_t> indices(N);

		RealType beta = 0.0;
		auto max_particle = std::max_element(particles.begin(), particles.end(),
				[](ParticleType a, ParticleType b) {
					return a.weight < b.weight;
				});
		RealType max_weight = (*max_particle).weight;
		for (size_t i = 0; i < N; i++) {
			beta += unif(rng) * 2.0 * max_weight;
			while (beta > particles[index].weight) {
				beta -= particles[index].weight;
				index = (index + 1) % N;
			}
			indices[i] = index;
		}
		return indices;
	}

	std::random_device rd;
	std::mt19937_64 rng;
	std::uniform_real_distribution<RealType> unif{0.0, 1.0};
};


template <size_t N,
	 typename ParticleType,
	 typename RealType = double,
	 typename Resampler = low_variance_resampler<N, ParticleType, RealType>>
struct particle_filter
{
	std::vector<ParticleType> particles;
	Resampler resampler;

	particle_filter() : particles(N) {}

	void move() {
		for(auto &p : particles)
			p.move();
	}

	void compute_weights(const typename ParticleType::mtype *Z) {
		for (auto &p : particles) {
			p.weight = p.compute_weight(Z);
		}
	}

	void resample() {
		std::vector<size_t> &&indices = resampler.resample(this->particles);
		std::vector<ParticleType> new_particles(N);
		for (size_t i = 0; i < N; i++)
			new_particles[i] = particles[indices[i]];
		std::swap(new_particles, particles);
	}
};


} // bayes::


#endif /* __PF_HPP__EA28733A_563C_4851_AC42_A55C2DE7A08C */

