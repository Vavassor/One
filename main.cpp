/* Written in 2017 by Andrew Dawson

To the extent possible under law, the author(s) have dedicated all copyright and
related and neighboring rights to this software to the public domain worldwide.
This software is distributed without any warranty.

You should have received a copy of the CC0 Public Domain Dedication along with
this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.

Build Instructions..............................................................

For playing, use these commands:

-for Windows, using the visual studio compiler
cl /O2 main.cpp kernel32.lib user32.lib gdi32.lib opengl32.lib /link /out:One.exe

-for Linux
g++ -o One -std=c++0x -O3 main.cpp -lGL -lX11

For debugging, use these commands:

-for Windows, using the visual studio compiler
cl /Od /Wall main.cpp kernel32.lib user32.lib gdi32.lib opengl32.lib /link /debug /out:One.exe

-for Linux
g++ -o One -std=c++0x -O0 -g3 -Wall -fmessage-length=0 main.cpp -lGL -lX11
*/

#if defined(__linux__)
#define OS_LINUX
#elif defined(_WIN32)
#define OS_WINDOWS
#else
#error Failed to figure out what operating system this is.
#endif

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

// Useful Things................................................................

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

#define ASSERT(expression) \
	assert(expression)

#define ALLOCATE(type, count) \
	static_cast<type*>(calloc((count), sizeof(type)))
#define REALLOCATE(memory, type, count) \
	static_cast<type*>(realloc((memory), sizeof(type) * (count)))
#define DEALLOCATE(memory) \
	free(memory)
#define SAFE_DEALLOCATE(memory) \
	if(memory) {DEALLOCATE(memory); (memory) = nullptr;}

static bool ensure_array_size(void** array, int* capacity, int item_size, int extra)
{
	while(extra >= *capacity)
	{
		int old_capacity = *capacity;
		if(*capacity == 0)
		{
			*capacity = 10;
		}
		else
		{
			*capacity *= 2;
		}
		void* new_array = realloc(*array, item_size * (*capacity));
		if(!new_array)
		{
			return false;
		}
		int size_changed = *capacity - old_capacity;
		u8* place = reinterpret_cast<u8*>(new_array) + item_size * old_capacity;
		memset(place, 0, item_size * size_changed);
		*array = new_array;
	}
	return true;
}

#define ENSURE_ARRAY_SIZE(array, extra) \
	ensure_array_size(reinterpret_cast<void**>(&array), &array##_capacity, sizeof(*(array)), array##_count + (extra))

#define ARRAY_COUNT(array) \
	static_cast<int>(sizeof(array) / sizeof(*(array)))

static int string_size(const char* string)
{
	ASSERT(string);
	const char* s;
	for(s = string; *s; ++s);
	return s - string;
}

#define TAU       6.28318530717958647692f
#define PI        3.14159265358979323846f
#define PI_OVER_2 1.57079632679489661923f

// Clock Function Declarations..................................................

struct Clock
{
	double frequency;
};

void initialise_clock(Clock* clock);
double get_time(Clock* clock);
void go_to_sleep(Clock* clock, double amount_to_sleep);

// Logging Function Declarations................................................

enum class LogLevel
{
	Debug,
	Error,
};

void log_add_message(LogLevel level, const char* format, ...);

#define LOG_ERROR(format, ...) \
	log_add_message(LogLevel::Error, format, ##__VA_ARGS__)

#if defined(NDEBUG)
#define LOG_DEBUG(format, ...) // do nothing
#else
#define LOG_DEBUG(format, ...) \
	log_add_message(LogLevel::Debug, format, ##__VA_ARGS__)
#endif

// Immediate Mode Drawing Declarations..........................................

struct Vector3;
struct AABB;

namespace immediate {

void draw();
void add_line(Vector3 start, Vector3 end, Vector3 colour);
void add_wire_aabb(AABB aabb, Vector3 colour);

} // namespace immediate

// Random Number Generation.....................................................

namespace arandom {

/*  Written in 2015 by Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright and
related and neighboring rights to this software to the public domain worldwide.
This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

u64 x;

static uint64_t splitmix64()
{
	uint64_t z = (x += UINT64_C(0x9E3779B97F4A7C15));
	z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
	z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
	return z ^ (z >> 31);
}

/*  Written in 2016 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright and
related and neighboring rights to this software to the public domain worldwide.
This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

u64 s[2];

static inline u64 rotl(const u64 x, int k)
{
	return (x << k) | (x >> (64 - k));
}

// Xoroshiro128+
u64 generate()
{
	const u64 s0 = s[0];
	u64 s1 = s[1];
	const u64 result = s0 + s1;

	s1 ^= s0;
	s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
	s[1] = rotl(s1, 36); // c

	return result;
}

// End of Blackman & Vigna's code

u64 seed(u64 value)
{
	u64 old_seed = x;
	x = value;
	s[0] = splitmix64();
	s[1] = splitmix64();
	return old_seed;
}

int int_range(int min, int max)
{
	int x = generate() % static_cast<u64>(max - min + 1);
	return min + x;
}

static inline float to_float(u64 x)
{
	union
	{
		u32 i;
		float f;
	} u;
	u.i = UINT32_C(0x7F) << 23 | x >> 41;
	return u.f - 1.0f;
}

float float_range(float min, float max)
{
	float f = to_float(generate());
	return min + f * (max - min);
}

} // namespace arandom

// Vector Functions.............................................................

#include <cmath>
#include <cfloat>

using std::abs;
using std::sqrt;
using std::isfinite;
using std::signbit;
using std::sin;
using std::cos;

struct Vector2
{
	float x, y;
};

struct Vector3
{
	float x, y, z;

	float& operator [] (int index) {return reinterpret_cast<float*>(this)[index];}
	const float& operator [] (int index) const {return reinterpret_cast<const float*>(this)[index];}
};

const Vector3 vector3_zero   = {0.0f, 0.0f, 0.0f};
const Vector3 vector3_one    = {1.0f, 1.0f, 1.0f};
const Vector3 vector3_unit_x = {1.0f, 0.0f, 0.0f};
const Vector3 vector3_unit_y = {0.0f, 1.0f, 0.0f};
const Vector3 vector3_unit_z = {0.0f, 0.0f, 1.0f};
const Vector3 vector3_min    = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
const Vector3 vector3_max    = {+FLT_MAX, +FLT_MAX, +FLT_MAX};

Vector3 operator + (Vector3 v0, Vector3 v1)
{
	Vector3 result;
	result.x = v0.x + v1.x;
	result.y = v0.y + v1.y;
	result.z = v0.z + v1.z;
	return result;
}

Vector3& operator += (Vector3& v0, Vector3 v1)
{
	v0.x += v1.x;
	v0.y += v1.y;
	v0.z += v1.z;
	return v0;
}

Vector3 operator - (Vector3 v0, Vector3 v1)
{
	Vector3 result;
	result.x = v0.x - v1.x;
	result.y = v0.y - v1.y;
	result.z = v0.z - v1.z;
	return result;
}

Vector3& operator -= (Vector3& v0, Vector3 v1)
{
	v0.x -= v1.x;
	v0.y -= v1.y;
	v0.z -= v1.z;
	return v0;
}

Vector3 operator * (Vector3 v, float s)
{
	Vector3 result;
	result.x = v.x * s;
	result.y = v.y * s;
	result.z = v.z * s;
	return result;
}

Vector3 operator * (float s, Vector3 v)
{
	Vector3 result;
	result.x = s * v.x;
	result.y = s * v.y;
	result.z = s * v.z;
	return result;
}

Vector3& operator *= (Vector3& v, float s)
{
	v.x *= s;
	v.y *= s;
	v.z *= s;
	return v;
}

Vector3 operator / (Vector3 v, float s)
{
	Vector3 result;
	result.x = v.x / s;
	result.y = v.y / s;
	result.z = v.z / s;
	return result;
}

Vector3& operator /= (Vector3& v, float s)
{
	v.x /= s;
	v.y /= s;
	v.z /= s;
	return v;
}

Vector3 operator - (Vector3 v)
{
	return {-v.x, -v.y, -v.z};
}

float squared_length(Vector3 v)
{
	return (v.x * v.x) + (v.y * v.y) + (v.z * v.z);
}

float length(Vector3 v)
{
	return sqrt(squared_length(v));
}

Vector3 normalise(Vector3 v)
{
	float l = length(v);
	ASSERT(l != 0.0f && isfinite(l));
	return v / l;
}

float dot(Vector3 v0, Vector3 v1)
{
	return (v0.x * v1.x) + (v0.y * v1.y) + (v0.z * v1.z);
}

Vector3 cross(Vector3 v0, Vector3 v1)
{
	Vector3 result;
	result.x = (v0.y * v1.z) - (v0.z * v1.y);
	result.y = (v0.z * v1.x) - (v0.x * v1.z);
	result.z = (v0.x * v1.y) - (v0.y * v1.x);
	return result;
}

// isotropic scale is just operator *
Vector3 anisotropic_scale(Vector3 v0, Vector3 v1)
{
	Vector3 result;
	result.x = v0.x * v1.x;
	result.y = v0.y * v1.y;
	result.z = v0.z * v1.z;
	return result;
}

Vector3 scale_reciprocal(Vector3 v0, Vector3 v1)
{
	Vector3 result;
	result.x = v0.x / v1.x;
	result.y = v0.y / v1.y;
	result.z = v0.z / v1.z;
	return result;
}

float lerp(float v0, float v1, float t)
{
	return (1.0f - t) * v0 + t * v1;
}

Vector3 lerp(Vector3 v0, Vector3 v1, float t)
{
	Vector3 result;
	result.x = lerp(v0.x, v1.x, t);
	result.y = lerp(v0.y, v1.y, t);
	result.z = lerp(v0.z, v1.z, t);
	return result;
}

Vector3 project(Vector3 a, Vector3 b)
{
    return (dot(a, b) / dot(b, b)) * b;
}

Vector3 reject(Vector3 a, Vector3 b)
{
    return a - project(a, b);
}

Vector3 orthogonal(Vector3 v)
{
	float l = length(v);
	float s = (v.x > 0.0f) ? l : -l;
	float xt = v.x + s;
	float d = -v.y / (s * xt);

	Vector3 result;
	result.x = d * xt;
	result.y = 1.0f + d * v.y;
	result.z = d * v.z;
	return result;
}

// Quaternion Functions.........................................................

static bool float_almost_one(float x)
{
	return abs(x - 1.0f) <= 0.0000005f;
}

struct Quaternion
{
	float w, x, y, z;
};

const Quaternion quaternion_identity = {1.0f, 0.0f, 0.0f, 0.0f};

float norm(Quaternion q)
{
	return sqrt((q.w * q.w) + (q.x * q.x) + (q.y * q.y) + (q.z * q.z));
}

Vector3 operator * (Quaternion q, Vector3 v)
{
    Vector3 vector_part = {q.x, q.y, q.z};
    Vector3 t = 2.0f * cross(vector_part, v);
    return v + (q.w * t) + cross(vector_part, t);
}

Quaternion axis_angle_rotation(Vector3 axis, float angle)
{
	ASSERT(isfinite(angle));

	angle /= 2.0f;
	float phase = sin(angle);
	Vector3 v = normalise(axis);

	Quaternion result;
	result.w = cos(angle);
	result.x = v.x * phase;
	result.y = v.y * phase;
	result.z = v.z * phase;
	// Rotations must be unit quaternions.
	ASSERT(float_almost_one(norm(result)));
	return result;
}

Quaternion rotation_to(Vector3 u, Vector3 v)
{
	Quaternion result;
	float k_cos_theta = dot(u, v);
	float k = sqrt(squared_length(u) * squared_length(v));
	if(float_almost_one(k_cos_theta / k))
	{
		result = axis_angle_rotation(orthogonal(u), PI);
	}
	else
	{
		result = axis_angle_rotation(cross(u, v), k_cos_theta + k);
	}
	return result;
}

// Matrix Functions.............................................................

struct Matrix4
{
	float elements[16]; // in row-major order

	float& operator [] (int index) {return elements[index];}
	const float& operator [] (int index) const {return elements[index];}
};

const Matrix4 matrix4_identity =
{{
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
}};

Matrix4 operator * (const Matrix4& a, const Matrix4& b)
{
	Matrix4 result;
	for(int i = 0; i < 4; ++i)
	{
		for(int j = 0; j < 4; ++j)
		{
			float m = 0.0f;
			for(int w = 0; w < 4; ++w)
			{
				 m += a[4 * i + w] * b[4 * w + j];
			}
			result[4 * i + j] = m;
		}
	}
	return result;
}

Vector3 operator * (const Matrix4& m, Vector3 v)
{
	float a = (m[12] * v.x) + (m[13] * v.y) + (m[14] * v.z) + m[15];

	Vector3 result;
	result.x = ((m[0] * v.x) + (m[1] * v.y) + (m[2]  * v.z) + m[3])  / a;
	result.y = ((m[4] * v.x) + (m[5] * v.y) + (m[6]  * v.z) + m[7])  / a;
	result.z = ((m[8] * v.x) + (m[9] * v.y) + (m[10] * v.z) + m[11]) / a;
	return result;
}

Matrix4 transpose(const Matrix4& m)
{
	return
	{{
		m[0], m[4], m[8],  m[12],
		m[1], m[5], m[9],  m[13],
		m[2], m[6], m[10], m[14],
		m[3], m[7], m[11], m[15]
	}};
}

Matrix4 view_matrix(Vector3 x_axis, Vector3 y_axis, Vector3 z_axis, Vector3 position)
{
	Matrix4 result;

	result[0]  = x_axis.x;
	result[1]  = x_axis.y;
	result[2]  = x_axis.z;
	result[3]  = -dot(x_axis, position);

	result[4]  = y_axis.x;
	result[5]  = y_axis.y;
	result[6]  = y_axis.z;
	result[7]  = -dot(y_axis, position);

	result[8]  = z_axis.x;
	result[9]  = z_axis.y;
	result[10] = z_axis.z;
	result[11] = -dot(z_axis, position);

	result[12] = 0.0f;
	result[13] = 0.0f;
	result[14] = 0.0f;
	result[15] = 1.0f;

	return result;
}

// This function assumes a right-handed coordinate system.
Matrix4 look_at_matrix(Vector3 position, Vector3 target, Vector3 world_up)
{
	Vector3 forward = normalise(position - target);
	Vector3 right = normalise(cross(world_up, forward));
	Vector3 up = normalise(cross(forward, right));
	return view_matrix(right, up, forward, position);
}

// This function assumes a right-handed coordinate system.
Matrix4 turn_matrix(Vector3 position, float yaw, float pitch, Vector3 world_up)
{
	Vector3 facing;
	facing.x = cos(pitch) * cos(yaw);
	facing.y = cos(pitch) * sin(yaw);
	facing.z = sin(pitch);
	Vector3 forward = normalise(facing);
	Vector3 right = normalise(cross(world_up, forward));
	Vector3 up = normalise(cross(forward, right));
	return view_matrix(right, up, forward, position);
}

// This transforms from a right-handed coordinate system to OpenGL's default
// clip space. A position will be viewable in this clip space if its x, y, and
// z components are in the range [-w,w] of its w component.
Matrix4 perspective_projection_matrix(float fovy, float width, float height, float near_plane, float far_plane)
{
	float coty = 1.0f / tan(fovy / 2.0f);
	float aspect_ratio = width / height;
	float neg_depth = near_plane - far_plane;

	Matrix4 result;

	result[0] = coty / aspect_ratio;
	result[1] = 0.0f;
	result[2] = 0.0f;
	result[3] = 0.0f;

	result[4] = 0.0f;
	result[5] = coty;
	result[6] = 0.0f;
	result[7] = 0.0f;

	result[8] = 0.0f;
	result[9] = 0.0f;
	result[10] = (near_plane + far_plane) / neg_depth;
	result[11] = 2.0f * near_plane * far_plane / neg_depth;

	result[12] = 0.0f;
	result[13] = 0.0f;
	result[14] = -1.0f;
	result[15] = 0.0f;

	return result;
}

Matrix4 compose_transform(Vector3 position, Quaternion orientation, Vector3 scale)
{
	float w = orientation.w;
	float x = orientation.x;
	float y = orientation.y;
	float z = orientation.z;

	float xw = x * w;
	float xx = x * x;
	float xy = x * y;
	float xz = x * z;

	float yw = y * w;
	float yy = y * y;
	float yz = y * z;

	float zw = z * w;
	float zz = z * z;

	Matrix4 result;

	result[0]  = (1.0f - 2.0f * (yy + zz)) * scale.x;
	result[1]  = (       2.0f * (xy - zw)) * scale.y;
	result[2]  = (       2.0f * (xz + yw)) * scale.z;
	result[3]  = position.x;

	result[4]  = (       2.0f * (xy + zw)) * scale.x;
	result[5]  = (1.0f - 2.0f * (xx + zz)) * scale.y;
	result[6]  = (       2.0f * (yz - xw)) * scale.z;
	result[7]  = position.y;

	result[8]  = (       2.0f * (xw - yw)) * scale.x;
	result[9]  = (       2.0f * (yz + xw)) * scale.y;
	result[10] = (1.0f - 2.0f * (xx + yy)) * scale.z;
	result[11] = position.z;

	result[12] = 0.0f;
	result[13] = 0.0f;
	result[14] = 0.0f;
	result[15] = 1.0f;

	return result;
}

Matrix4 inverse_transform(const Matrix4& m)
{
	// The scale can be extracted from the rotation data by just taking the
	// length of the first three row vectors.

	float dx = sqrt(m[0] * m[0] + m[1] * m[1] + m[2]  * m[2]);
	float dy = sqrt(m[4] * m[4] + m[5] * m[5] + m[6]  * m[6]);
	float dz = sqrt(m[8] * m[8] + m[9] * m[9] + m[10] * m[10]);

	// The extracted scale can then be divided out to isolate the rotation rows.

	float m00 = m[0] / dx;
	float m10 = m[4] / dy;
	float m20 = m[8] / dz;

	float m01 = m[1] / dx;
	float m11 = m[5] / dy;
	float m21 = m[9] / dz;

	float m02 = m[2] / dx;
	float m12 = m[6] / dy;
	float m22 = m[10] / dz;

	// The inverse of the translation elements is the negation of the
	// translation vector multiplied by the transpose of the rotation and the
	// reciprocal of the dilation.

	float a = -(m00 * m[3] + m10 * m[7] + m20 * m[11]) / dx;
	float b = -(m01 * m[3] + m11 * m[7] + m21 * m[11]) / dy;
	float c = -(m02 * m[3] + m12 * m[7] + m22 * m[11]) / dz;

	// After the unmodified rotation elements have been used to figure out the
	// inverse translation, they can be modified with the inverse dilation.

	m00 /= dx;
	m11 /= dy;
	m22 /= dz;

	// Put everything in, making sure to place the rotation elements in
	// transposed order.

	return
	{{
		m00, m10, m20, a,
		m01, m11, m21, b,
		m02, m12, m22, c,
		0.0f, 0.0f, 0.0f, 1.0f
	}};
}

// Collision Functions..........................................................

struct Triangle
{
	Vector3 vertices[3];
};

// BIH Tree Functions...........................................................

// Axis-Aligned Bounding Box
struct AABB
{
	Vector3 min;
	Vector3 max;
};

static bool aabb_overlap(AABB b0, AABB b1)
{
	Vector3 e0 = (b0.max - b0.min) / 2.0f;
	Vector3 e1 = (b1.max - b1.min) / 2.0f;
	Vector3 c0 = b0.min + e0;
	Vector3 c1 = b1.min + e1;
	Vector3 t = c1 - c0;
	return abs(t.x) <= e0.x + e1.x
		&& abs(t.y) <= e0.y + e1.y
		&& abs(t.z) <= e0.z + e1.z;
}

static bool aabb_contains(AABB outer, AABB inner)
{
	return inner.max.x <= outer.max.x
		&& inner.max.y <= outer.max.y
		&& inner.max.z <= outer.max.z
		&& inner.min.x >= outer.min.x
		&& inner.min.y >= outer.min.y
		&& inner.min.z >= outer.min.z;
}

static bool aabb_validate(AABB b)
{
	return b.max.x > b.min.x
		&& b.max.y > b.min.y
		&& b.max.z > b.min.z;
}

static AABB aabb_from_triangle(Triangle* triangle)
{
	AABB result;
	Vector3 v0 = triangle->vertices[0];
	Vector3 v1 = triangle->vertices[1];
	Vector3 v2 = triangle->vertices[2];
	result.min.x = fmin(fmin(v0.x, v1.x), v2.x);
	result.min.y = fmin(fmin(v0.y, v1.y), v2.y);
	result.min.z = fmin(fmin(v0.z, v1.z), v2.z);
	result.max.x = fmax(fmax(v0.x, v1.x), v2.x);
	result.max.y = fmax(fmax(v0.y, v1.y), v2.y);
	result.max.z = fmax(fmax(v0.z, v1.z), v2.z);
	return result;
}

static AABB aabb_from_ellipsoid(Vector3 center, Vector3 radius)
{
	AABB result;
	result.min = center - radius;
	result.max = center + radius;
	return result;
}

static AABB aabb_merge(AABB o0, AABB o1)
{
	AABB result;
	result.min.x = fmin(o0.min.x, o1.min.x);
	result.min.y = fmin(o0.min.y, o1.min.y);
	result.min.z = fmin(o0.min.z, o1.min.z);
	result.max.x = fmax(o0.max.x, o1.max.x);
	result.max.y = fmax(o0.max.y, o1.max.y);
	result.max.z = fmax(o0.max.z, o1.max.z);
	return result;
}

static AABB compute_bounds(Triangle* triangles, int lower, int upper)
{
	AABB result = {vector3_max, vector3_min};
	for(int i = lower; i <= upper; ++i)
	{
		AABB aabb = aabb_from_triangle(triangles + i);
		result = aabb_merge(result, aabb);
	}
	return result;
}

static bool aabb_intersect(AABB a, Vector3 a_velocity, AABB b, Vector3 b_velocity, float* t0, float* t1)
{
	if(aabb_overlap(a, b))
	{
		*t0 = 0.0f;
		*t1 = 0.0f;
		return true;
	}

	Vector3 u0 = vector3_min;
	Vector3 u1 = vector3_max;
	Vector3 v = b_velocity - a_velocity;
	for(int i = 0; i < 3; ++i)
	{
		if(a.max[i] < b.min[i] && v[i] < 0.0f)
		{
			u0[i] = (a.max[i] - b.min[i]) / v[i];
		}
		else if(b.max[i] < a.min[i] && v[i] > 0.0f)
		{
			u0[i] = (a.min[i] - b.max[i]) / v[i];
		}
		if(b.max[i] > a.min[i] && v[i] < 0.0f)
		{
			u1[i] = (a.min[i] - b.max[i]) / v[i];
		}
		else if(a.max[i] > b.min[i] && v[i] > 0.0f)
		{
			u1[i] = (a.max[i] - b.min[i]) / v[i];
		}
	}
	*t0 = fmax(u0[0], fmax(u0[1], u0[2]));
	*t1 = fmin(u1[0], fmin(u1[1], u1[2]));

	bool result = *t0 >= 0.0f && *t1 <= 1.0f && *t0 <= *t1;
	if(result)
	{
		return result;
	}

	return result;
}

// Bounding Interval Hierarchy
namespace bih {

enum Flag
{
	FLAG_X,
	FLAG_Y,
	FLAG_Z,
	FLAG_LEAF,
};

static Flag get_longest_axis(AABB o)
{
	Flag result;
	float x = o.max.x - o.min.x;
	float y = o.max.y - o.min.y;
	float z = o.max.z - o.min.z;
	if(y > x)
	{
		if(z > y)
		{
			result = FLAG_Z;
		}
		else
		{
			result = FLAG_Y;
		}
	}
	else
	{
		if(z > x)
		{
			result = FLAG_Z;
		}
		else
		{
			result = FLAG_X;
		}
	}
	ASSERT(result >= 0 && result < 3);
	return result;
}

struct Node
{
	union
	{
		float clip[2];
		u32 items; // used only by leaf nodes
	};
	u32 index : 30;
	Flag flag : 2;
};

struct Tree
{
	AABB bounds;
	Node* nodes;
	int nodes_count;
	int nodes_capacity;
};

static int allocate_nodes(Tree* tree, int count)
{
	bool success = ENSURE_ARRAY_SIZE(tree->nodes, count);
	if(!success)
	{
		return -1;
	}
	// Only 30 bits are used to store children indices in each node.
	ASSERT(tree->nodes_count <= 0x3fffffff);
	if(tree->nodes_count > 0x3fffffff)
	{
		return -1;
	}
	int result = tree->nodes_count;
	tree->nodes_count += count;
	return result;
}

static float compute_just_min(Triangle* triangles, int lower, int upper, Flag axis)
{
	float result = FLT_MAX;
	for(int i = lower; i <= upper; ++i)
	{
		Triangle* triangle = triangles + i;
		float v0 = triangle->vertices[0][axis];
		float v1 = triangle->vertices[1][axis];
		float v2 = triangle->vertices[2][axis];
		float min = fmin(fmin(v0, v1), v2);
		result = fmin(result, min);
	}
	return result;
}

static float compute_just_max(Triangle* triangles, int lower, int upper, Flag axis)
{
	float result = -FLT_MAX;
	for(int i = lower; i <= upper; ++i)
	{
		Triangle* triangle = triangles + i;
		float v0 = triangle->vertices[0][axis];
		float v1 = triangle->vertices[1][axis];
		float v2 = triangle->vertices[2][axis];
		float max = fmax(fmax(v0, v1), v2);
		result = fmax(result, max);
	}
	return result;
}

static int partition_triangles(Triangle* triangles, int lower, int upper, float split, Flag axis)
{
	int pivot = lower;
	int j = upper;
	while(pivot <= j)
	{
		Triangle* triangle = triangles + pivot;
		float v0 = triangle->vertices[0][axis];
		float v1 = triangle->vertices[1][axis];
		float v2 = triangle->vertices[2][axis];
		float a = (v0 + v1 + v2) / 3.0f;
		if(a > split)
		{
			Triangle temp = triangles[pivot];
			triangles[pivot] = triangles[j];
			triangles[j] = temp;
			j -= 1;
		}
		else
		{
			pivot += 1;
		}
	}
	if(pivot == lower && j < pivot)
	{
		pivot = j;
	}
	return pivot;
}

bool build_node(Tree* tree, Node* node, AABB bounds, Triangle* triangles, int lower, int upper)
{
	ASSERT(aabb_validate(bounds));
	ASSERT(lower <= upper);

	if(upper - lower < 1)
	{
		node->flag = FLAG_LEAF;
		node->items = lower;
		return true;
	}

	AABB aabb = compute_bounds(triangles, lower, upper);
#if 0
	Flag axis = get_longest_axis(aabb);
#else
	Vector3 e0 = (bounds.max - bounds.min) / 2.0f;
	Vector3 e1 = (aabb.max - aabb.min) / 2.0f;
	Vector3 e = e0 - e1;
	Flag axis;
	if(e.y > e.x)
	{
		if(e.z > e.y)
		{
			axis = FLAG_Z;
		}
		else
		{
			axis = FLAG_Y;
		}
	}
	else
	{
		if(e.z > e.x)
		{
			axis = FLAG_Z;
		}
		else
		{
			axis = FLAG_X;
		}
	}
#endif

	float split = (aabb.max[axis] + aabb.min[axis]) / 2.0f;
	int pivot = partition_triangles(triangles, lower, upper, split, axis);

	if(pivot < lower)
	{
		AABB right_bounds = aabb;
		right_bounds.min[axis] = split;
		return build_node(tree, node, right_bounds, triangles, lower, upper);
	}
	else if(pivot > upper)
	{
		AABB left_bounds = aabb;
		left_bounds.max[axis] = split;
		return build_node(tree, node, left_bounds, triangles, lower, upper);
	}
	else
	{
		node->flag = axis;

		int index = allocate_nodes(tree, 2);
		if(index < 0)
		{
			return false;
		}
		node->index = index;
		Node* children = tree->nodes + index;

		Node* child_left = children + 0;
		int left_upper = fmax(lower, pivot - 1);
		AABB left_bounds = aabb;
		left_bounds.max[axis] = split;
		node->clip[0] = compute_just_max(triangles, lower, left_upper, axis);
		bool built0 = build_node(tree, child_left, left_bounds, triangles, lower, left_upper);

		Node* child_right = children + 1;
		AABB right_bounds = aabb;
		right_bounds.min[axis] = split;
		node->clip[1] = compute_just_min(triangles, pivot, upper, axis);
		bool built1 = build_node(tree, child_right, right_bounds, triangles, pivot, upper);

		return built0 && built1;
	}
}

bool build_tree(Tree* tree, Triangle* triangles, int triangles_count)
{
	// 2 * triangles_count is just an arbitrary estimate.
	int nodes_capacity = 2 * triangles_count;
	Node* nodes = ALLOCATE(Node, nodes_capacity);
	if(!nodes)
	{
		return false;
	}
	tree->nodes = nodes;
	tree->nodes_capacity = nodes_capacity;

	int index = allocate_nodes(tree, 1);
	if(index < 0)
	{
		return false;
	}
	Node* root = tree->nodes + index;
	tree->bounds = compute_bounds(triangles, 0, triangles_count - 1);

	return build_node(tree, root, tree->bounds, triangles, 0, triangles_count - 1);
}

struct IntersectionResult
{
	int* indices;
	int indices_count;
	int indices_capacity;
};

bool intersect_node(Node* nodes, Node* node, AABB node_bounds, AABB aabb, Vector3 velocity, IntersectionResult* result)
{
	ASSERT(aabb_validate(node_bounds));
	ASSERT(aabb_validate(aabb));

	float t0, t1;
	bool intersects = aabb_intersect(aabb, velocity, node_bounds, vector3_zero, &t0, &t1);
	if(!intersects)
	{
		return false;
	}

	if(node->flag == bih::FLAG_LEAF)
	{
		bool big_enough = ENSURE_ARRAY_SIZE(result->indices, 1);
		if(!big_enough)
		{
			return false;
		}
		result->indices[result->indices_count] = node->items;
		result->indices_count += 1;
		return true;
	}

	Flag axis = node->flag;
	Node* children = nodes + node->index;

	Node* left = children + 0;
	AABB left_bounds = node_bounds;
	left_bounds.max[axis] = node->clip[0];
	bool intersects0 = intersect_node(nodes, left, left_bounds, aabb, velocity, result);

	Node* right = children + 1;
	AABB right_bounds = node_bounds;
	right_bounds.min[axis] = node->clip[1];
	bool intersects1 = intersect_node(nodes, right, right_bounds, aabb, velocity, result);

	return intersects0 || intersects1;
}

bool intersect_tree(Tree* tree, AABB aabb, Vector3 velocity, IntersectionResult* result)
{
	return intersect_node(tree->nodes, tree->nodes, tree->bounds, aabb, velocity, result);
}

} // namespace bih

// Collision Functions..........................................................

struct CollisionPacket
{
	Vector3 intersection_point;
	float nearest_distance;
	bool found_collision;
};

float clamp(float a, float min, float max)
{
	return fmin(fmax(a, min), max);
}

bool point_in_triangle(Vector3 point, Vector3 pa, Vector3 pb, Vector3 pc)
{
	Vector3 e10 = pb - pa;
	Vector3 e20 = pc - pa;

	float a = dot(e10, e10);
	float b = dot(e10, e20);
	float c = dot(e20, e20);
	float ac_bb = (a * c) - (b * b);
	Vector3 vp = point - pa;

	float d = dot(vp, e10);
	float e = dot(vp, e20);
	float x = (d * c) - (e * b);
	float y = (e * a) - (d * b);
	float z = x + y - ac_bb;

	return signbit(z) && !(signbit(x) || signbit(y));
}

bool get_lowest_root(float a, float b, float c, float max_r, float* root)
{
	// The roots of the quadratic equation ax² + bx + c = 0 can be obtained with
	// the formula (-b ± √(b² - 4ac)) / 2a.
	float d = (b * b) - (4.0f * a * c);
	if(d < 0.0f)
	{
		// The two roots are complex.
		return false;
	}
	// Calculate the two real roots.
	float sd = sqrt(d);
	float r1 = (-b - sd) / (2.0f * a);
	float r2 = (-b + sd) / (2.0f * a);
	// Determine which is smaller, positive, and below the given limit.
	if(r1 > r2)
	{
		float temp = r2;
		r2 = r1;
		r1 = temp;
	}
	if(r1 > 0.0f && r1 < max_r)
	{
		*root = r1;
		return true;
	}
	else if(r2 > 0.0f && r2 < max_r)
	{
		*root = r2;
		return true;
	}
	else
	{
		return false;
	}
}

// signed distance from a point to a plane
float signed_distance(Vector3 point, Vector3 origin, Vector3 normal)
{
	return dot(point, normal) - dot(origin, normal);
}

void set_packet(CollisionPacket* packet, Vector3 collision_point, float t, Vector3 velocity)
{
	float distance = t * length(velocity);
	if(!packet->found_collision || distance < packet->nearest_distance)
	{
		packet->nearest_distance = distance;
		packet->intersection_point = collision_point;
		packet->found_collision = true;
	}
}

void collide_unit_sphere_with_triangle(Vector3 center, Vector3 velocity, Triangle triangle, CollisionPacket* packet)
{
	Vector3 p1 = triangle.vertices[0];
	Vector3 p2 = triangle.vertices[1];
	Vector3 p3 = triangle.vertices[2];
	Vector3 normal = normalise(cross(p2 - p1, p3 - p1));

	float normal_dot_velocity = dot(normal, velocity);
	float signed_distance_to_plane = signed_distance(center, p1, normal);

	if(normal_dot_velocity > 0.0f)
	{
		// The sphere is moving away from the surface, or oriented opposite it.
		return;
	}
	else if(normal_dot_velocity == 0.0f)
	{
		// Is the sphere traveling parallel to the plane and far away enough to not
		// touch it?
		if(abs(signed_distance_to_plane) >= 1.0f)
		{
			return;
		}
	}
	else
	{
		// The sphere must be traveling toward the plane. Try intersecting the range
		// of its sweep, because if that's not within the plane, then it may not
		// collide.
		float t0 = (-1.0f - signed_distance_to_plane) / normal_dot_velocity;
		float t1 = (+1.0f - signed_distance_to_plane) / normal_dot_velocity;
		// Swap so that t0 < t1.
		if(t0 > t1)
		{
			float temp = t1;
			t1 = t0;
			t0 = temp;
		}
		// Check that at least one result is within the range.
		if(t0 > 1.0f || t1 < 0.0f)
		{
			return;
		}
		t0 = clamp(t0, 0.0f, 1.0f);
		// t1 = clamp(t1, 0.0f, 1.0f); // This isn't actually used.

		// The sweep must intersect the triangle's plane.
		// Test if the center of the sweep goes through the face of the triangle.
		Vector3 intersection = (center - normal) + (t0 * velocity);
		if(point_in_triangle(intersection, p1, p2, p3))
		{
			set_packet(packet, intersection, t0, velocity);
			return;
		}
	}

	// At this point, some part of the plane must be hit apart from the triangle
	// face just ruled out.
	// Test each vertex and edge of the triangle.
	bool found_collision = false;
	float t = 1.0f;
	Vector3 collision_point;

	auto sweep_point = [center, velocity, &found_collision, &t, &collision_point](Vector3 p)
	{
		// Calculate the parameters of the equation at² + bt + c = 0.
		float a = squared_length(velocity);
		float b = 2.0f * dot(velocity, center - p);
		float c = squared_length(p - center) - 1.0f;
		float new_t;
		if(get_lowest_root(a, b, c, t, &new_t))
		{
			found_collision = true;
			t = new_t;
			collision_point = p;
		}
	};

	sweep_point(p1);
	sweep_point(p2);
	sweep_point(p3);

	auto sweep_edge = [center, velocity, &found_collision, &t, &collision_point](Vector3 e1, Vector3 e2)
	{
		Vector3 edge = e2 - e1;
		Vector3 base_to_vertex = e1 - center;
		float el = squared_length(edge);
		float ev = dot(edge, velocity);
		float eb = dot(edge, base_to_vertex);
		float a = el * -squared_length(velocity) + ev * ev;
		float b = el * (2.0f * dot(velocity, base_to_vertex)) - (2.0f * ev * eb);
		float c = el * (1.0f - squared_length(base_to_vertex)) + (eb * eb);
		float new_t;
		// Does the swept sphere collide against the line of the edge?
		if(get_lowest_root(a, b, c, t, &new_t))
		{
			// Check if the intersection is within the line segment.
			float f = (ev * new_t - eb) / el;
			if(f >= 0.0 && f <= 1.0)
			{
				found_collision = true;
				t = new_t;
				collision_point = e1 + (f * edge);
			}
		}
	};

	sweep_edge(p1, p2);
	sweep_edge(p2, p3);
	sweep_edge(p3, p1);

	if(found_collision)
	{
		set_packet(packet, collision_point, t, velocity);
	}
}

struct World
{
	bih::Tree tree;
	Triangle* triangles;
	int triangles_count;
};

void check_collision(Vector3 position, Vector3 radius, Vector3 velocity, World* world, CollisionPacket* packet)
{
	Vector3 world_position = anisotropic_scale(position, radius);
	Vector3 world_velocity = anisotropic_scale(velocity, radius);
	AABB aabb = aabb_from_ellipsoid(world_position, radius);
	bih::IntersectionResult intersection = {};
	bih::intersect_tree(&world->tree, aabb, world_velocity, &intersection);
	for(int i = 0; i < intersection.indices_count; ++i)
	{
		int index = intersection.indices[i];
		Triangle triangle = world->triangles[index];
		// Transform the triangle's vertex positions to ellipsoid space.
		for(int j = 0; j < 3; ++j)
		{
			Vector3 v = triangle.vertices[j];
			v = scale_reciprocal(v, radius);
			triangle.vertices[j] = v;
		}
		collide_unit_sphere_with_triangle(position, velocity, triangle, packet);
#if 0
		// Debug draw each triangle that is checked against.
		// immediate::draw() cannot be called here, so just buffer the
		// primitives and hope they're drawn during the render cycle.
		{
			triangle = world->triangles[index];
			Vector3 offset = {0.0f, 0.0f, 0.03f};
			Vector3 v0 = triangle.vertices[0] + offset;
			Vector3 v1 = triangle.vertices[1] + offset;
			Vector3 v2 = triangle.vertices[2] + offset;
			Vector3 center = (v0 + v1 + v2) / 3.0f;
			Vector3 outside_colour = {1.0f, 1.0f, 1.0f};
			Vector3 center_colour = {0.85f, 0.85f, 0.85f};
			immediate::add_line(v0, v1, outside_colour);
			immediate::add_line(v1, v2, outside_colour);
			immediate::add_line(v2, v0, outside_colour);
			immediate::add_line(center, v0, center_colour);
			immediate::add_line(center, v1, center_colour);
			immediate::add_line(center, v2, center_colour);
		}
#endif
	}
	SAFE_DEALLOCATE(intersection.indices);
}

Vector3 collide_with_world(Vector3 position, Vector3 radius, Vector3 velocity, World* world)
{
	CollisionPacket go = {};
	CollisionPacket* packet = &go;

	const float very_close = 0.0005f;
	const int iteration_limit = 5;
	for(int i = 0; i < iteration_limit; ++i)
	{
		packet->found_collision = false;
		check_collision(position, radius, velocity, world, packet);

		// Just move freely if no collision is found.
		if(!packet->found_collision)
		{
			return position + velocity;
		}

		// Only update if we are not already very close. And if so, only move
		// very close to intersection, not to the exact spot.
		Vector3 new_position = position;
		if(packet->nearest_distance >= very_close)
		{
			Vector3 v = normalise(velocity);
			new_position = position + (packet->nearest_distance - very_close) * v;
			// Adjust the intersection point, so the sliding plane will be
			// unaffected by the fact that we move slightly less than collision
			// tells us.
			packet->intersection_point -= very_close * v;
		}

		// Calculate the outgoing velocity along the slide plane.
		Vector3 d = position + velocity; // destination
		Vector3 po = packet->intersection_point; // plane origin
		Vector3 pn = normalise(new_position - po); // plane normal
		Vector3 new_destination = d - (signed_distance(d, po, pn) * pn);
		Vector3 new_velocity = new_destination - packet->intersection_point;

		// Ignore the remaining velocity if it becomes negligible.
		if(length(new_velocity) < very_close)
		{
			return new_position;
		}

		if(velocity.x == new_velocity.x && velocity.y == new_velocity.y && velocity.z == new_velocity.z)
		{
			position.x = new_position.x;
		}

		position = new_position;
		velocity = new_velocity;
	}

	// Supposing there's so many simultaneous collisions that it reaches the
	// iteration limit, it's not unreasonable to assume no movement.
	return position;
}

Vector3 collide_and_slide(Vector3 position, Vector3 radius, Vector3 velocity, Vector3 gravity, World* world)
{
	Vector3 e_position = scale_reciprocal(position, radius);
	Vector3 e_velocity = scale_reciprocal(velocity, radius);
	Vector3 collide_position;
	if(squared_length(e_velocity) != 0.0f)
	{
		collide_position = collide_with_world(e_position, radius, e_velocity, world);
		position = anisotropic_scale(collide_position, radius);
		e_position = collide_position;
	}
	velocity = gravity;

	e_velocity = scale_reciprocal(gravity, radius);
	collide_position = collide_with_world(e_position, radius, e_velocity, world);
	Vector3 final_position = anisotropic_scale(collide_position, radius);

	return final_position;
}

// OpenGL Function and Type Declarations........................................

#if defined(__gl_h_) || defined(__GL_H__)
#error gl.h included before this section
#endif
#if defined(__gltypes_h_)
#error gltypes.h included before this section
#endif

#define __gl_h_
#define __GL_H__
#define __gltypes_h_

typedef unsigned int GLenum;
typedef unsigned char GLboolean;
typedef unsigned int GLbitfield;
typedef void GLvoid;
typedef signed char GLbyte;
typedef short GLshort;
typedef int GLint;
typedef unsigned char GLubyte;
typedef unsigned short GLushort;
typedef unsigned int GLuint;
typedef int GLsizei;
typedef float GLfloat;
typedef float GLclampf;
typedef double GLdouble;
typedef double GLclampd;
typedef char GLchar;
typedef char GLcharARB;
typedef ptrdiff_t GLsizeiptr;

#define GL_COLOR_BUFFER_BIT 0x00004000
#define GL_CULL_FACE 0x0B44
#define GL_DEPTH_BUFFER_BIT 0x00000100
#define GL_DEPTH_TEST 0x0B71
#define GL_FALSE 0
#define GL_FLOAT 0x1406
#define GL_LINES 0x0001
#define GL_TRIANGLES 0x0004
#define GL_TRUE 1
#define GL_UNSIGNED_BYTE 0x1401
#define GL_UNSIGNED_INT 0x1405
#define GL_UNSIGNED_SHORT 0x1403

#define GL_ARRAY_BUFFER 0x8892
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#define GL_STATIC_DRAW 0x88E4
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_WRITE_ONLY 0x88B9

#define GL_COMPILE_STATUS 0x8B81
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_LINK_STATUS 0x8B82
#define GL_VERTEX_SHADER 0x8B31

#if defined(OS_LINUX)
#define APIENTRYA

#elif defined(OS_WINDOWS)
#if defined(__MINGW32__) && !defined(_ARM_)
#define APIENTRY __stdcall
#elif (_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED)
#define APIENTRY __stdcall
#else
#define APIENTRY
#endif
#define APIENTRYA APIENTRY
#endif

void (APIENTRYA *p_glClear)(GLbitfield mask) = nullptr;
void (APIENTRYA *p_glEnable)(GLenum cap) = nullptr;
void (APIENTRYA *p_glViewport)(GLint x, GLint y, GLsizei width, GLsizei height) = nullptr;

void (APIENTRYA *p_glDrawArrays)(GLenum mode, GLint first, GLsizei count) = nullptr;
void (APIENTRYA *p_glDrawElements)(GLenum mode, GLsizei count, GLenum type, const void* indices) = nullptr;

void (APIENTRYA *p_glBindVertexArray)(GLuint ren_array) = nullptr;
void (APIENTRYA *p_glDeleteVertexArrays)(GLsizei n, const GLuint* arrays) = nullptr;
void (APIENTRYA *p_glGenVertexArrays)(GLsizei n, GLuint* arrays) = nullptr;

void (APIENTRYA *p_glBindBuffer)(GLenum target, GLuint buffer) = nullptr;
void (APIENTRYA *p_glBufferData)(GLenum target, GLsizeiptr size, const void* data, GLenum usage) = nullptr;
void (APIENTRYA *p_glDeleteBuffers)(GLsizei n, const GLuint* buffers) = nullptr;
void (APIENTRYA *p_glGenBuffers)(GLsizei n, GLuint* buffers) = nullptr;
void* (APIENTRYA *p_glMapBuffer)(GLenum buffer, GLenum access) = nullptr;
GLboolean (APIENTRYA *p_glUnmapBuffer)(GLenum target) = nullptr;

void (APIENTRYA *p_glAttachShader)(GLuint program, GLuint shader) = nullptr;
void (APIENTRYA *p_glCompileShader)(GLuint shader) = nullptr;
GLuint (APIENTRYA *p_glCreateProgram)(void) = nullptr;
GLuint (APIENTRYA *p_glCreateShader)(GLenum type) = nullptr;
void (APIENTRYA *p_glDeleteProgram)(GLuint program) = nullptr;
void (APIENTRYA *p_glDeleteShader)(GLuint shader) = nullptr;
void (APIENTRYA *p_glDetachShader)(GLuint program, GLuint shader) = nullptr;
void (APIENTRYA *p_glEnableVertexAttribArray)(GLuint index) = nullptr;
void (APIENTRYA *p_glGetProgramInfoLog)(GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog) = nullptr;
void (APIENTRYA *p_glGetProgramiv)(GLuint program, GLenum pname, GLint* params) = nullptr;
void (APIENTRYA *p_glGetShaderInfoLog)(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog) = nullptr;
void (APIENTRYA *p_glGetShaderiv)(GLuint shader, GLenum pname, GLint* params) = nullptr;
GLint (APIENTRYA *p_glGetUniformLocation)(GLuint program, const GLchar* name) = nullptr;
void (APIENTRYA *p_glLinkProgram)(GLuint program) = nullptr;
void (APIENTRYA *p_glShaderSource)(GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length) = nullptr;
void (APIENTRYA *p_glUniform3fv)(GLint location, GLsizei count, const GLfloat* value) = nullptr;
void (APIENTRYA *p_glUniformMatrix4fv)(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value) = nullptr;
void (APIENTRYA *p_glUseProgram)(GLuint program) = nullptr;
void (APIENTRYA *p_glVertexAttribPointer)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer) = nullptr;

#define glBindVertexArray p_glBindVertexArray
#define glDeleteVertexArrays p_glDeleteVertexArrays
#define glGenVertexArrays p_glGenVertexArrays

#define glBindBuffer p_glBindBuffer
#define glBufferData p_glBufferData
#define glDeleteBuffers p_glDeleteBuffers
#define glGenBuffers p_glGenBuffers
#define glMapBuffer p_glMapBuffer
#define glUnmapBuffer p_glUnmapBuffer

#define glClear p_glClear
#define glEnable p_glEnable
#define glViewport p_glViewport

#define glDrawArrays p_glDrawArrays
#define glDrawElements p_glDrawElements

#define glAttachShader p_glAttachShader
#define glCompileShader p_glCompileShader
#define glCreateProgram p_glCreateProgram
#define glCreateShader p_glCreateShader
#define glDeleteProgram p_glDeleteProgram
#define glDeleteShader p_glDeleteShader
#define glDetachShader p_glDetachShader
#define glEnableVertexAttribArray p_glEnableVertexAttribArray
#define glGetProgramInfoLog p_glGetProgramInfoLog
#define glGetProgramiv p_glGetProgramiv
#define glGetShaderInfoLog p_glGetShaderInfoLog
#define glGetShaderiv p_glGetShaderiv
#define glGetUniformLocation p_glGetUniformLocation
#define glLinkProgram p_glLinkProgram
#define glShaderSource p_glShaderSource
#define glUniform3fv p_glUniform3fv
#define glUniformMatrix4fv p_glUniformMatrix4fv
#define glUseProgram p_glUseProgram
#define glVertexAttribPointer p_glVertexAttribPointer

// Shader Functions.............................................................

static GLuint load_shader(GLenum type, const char* source)
{
	GLuint shader = glCreateShader(type);
	GLint source_size = string_size(source);
	glShaderSource(shader, 1, &source, &source_size);
	glCompileShader(shader);

	// Output any errors if the compilation failed.

	GLint compile_status = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
	if(compile_status == GL_FALSE)
	{
		GLint info_log_size = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_size);
		if(info_log_size > 0)
		{
			GLchar* info_log = ALLOCATE(GLchar, info_log_size);
			if(info_log)
			{
				GLsizei bytes_written = 0;
				glGetShaderInfoLog(shader, info_log_size, &bytes_written, info_log);
				LOG_ERROR("Couldn't compile the shader.\n%s", info_log);
				DEALLOCATE(info_log);
			}
			else
			{
				LOG_ERROR("Couldn't compile the shader.");
			}
		}

		glDeleteShader(shader);

		return 0;
	}

	return shader;
}

static GLuint load_shader_program(const char* vertex_source, const char* fragment_source)
{
	GLuint program;

	GLuint vertex_shader = load_shader(GL_VERTEX_SHADER, vertex_source);
	if(vertex_shader == 0)
	{
		LOG_ERROR("Failed to load the vertex shader.");
		return 0;
	}

	GLuint fragment_shader = load_shader(GL_FRAGMENT_SHADER, fragment_source);
	if(fragment_shader == 0)
	{
		LOG_ERROR("Failed to load the fragment shader.");
		glDeleteShader(vertex_shader);
		return 0;
	}

	// Create the program object and link the shaders to it.

	program = glCreateProgram();
	glAttachShader(program, vertex_shader);
	glAttachShader(program, fragment_shader);
	glLinkProgram(program);

	// Check if linking failed and output any errors.

	GLint link_status = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &link_status);
	if(link_status == GL_FALSE)
	{
		int info_log_size = 0;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_size);
		if(info_log_size > 0)
		{
			GLchar* info_log = ALLOCATE(GLchar, info_log_size);
			if(info_log)
			{
				GLsizei bytes_written = 0;
				glGetProgramInfoLog(program, info_log_size, &bytes_written, info_log);
				LOG_ERROR("Couldn't link the shader program.\n%s", info_log);
				DEALLOCATE(info_log);
			}
			else
			{
				LOG_ERROR("Couldn't link the shader program.");
			}
		}

		glDeleteProgram(program);
		glDeleteShader(fragment_shader);
		glDeleteShader(vertex_shader);

		return 0;
	}

	// Shaders are no longer needed after the program object is linked.
	glDetachShader(program, vertex_shader);
	glDetachShader(program, fragment_shader);

	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);

	return program;
}

// Floor Functions..............................................................

struct VertexPC
{
	Vector3 position;
	Vector3 colour;
};

struct VertexPNC
{
	Vector3 position;
	Vector3 normal;
	Vector3 colour;
};

struct Floor
{
	VertexPNC* vertices;
	u16* indices;
	int vertices_capacity;
	int vertices_count;
	int indices_capacity;
	int indices_count;
};

static void floor_destroy(Floor* floor)
{
	SAFE_DEALLOCATE(floor->vertices);
	SAFE_DEALLOCATE(floor->indices);
}

// This box only has 4 faces.
static bool floor_add_box(Floor* floor, Vector3 bottom_left, Vector3 dimensions)
{
	bool resized0 = ENSURE_ARRAY_SIZE(floor->vertices, 16);
	bool resized1 = ENSURE_ARRAY_SIZE(floor->indices, 24);
	if(!resized0 || !resized1)
	{
		return false;
	}

	float l = bottom_left.x;                // left
	float n = bottom_left.y;                // near
	float b = bottom_left.z;                // bottom
	float r = bottom_left.x + dimensions.x; // right
	float f = bottom_left.y + dimensions.y; // far
	float t = bottom_left.z + dimensions.z; // top

	int o = floor->vertices_count;
	floor->vertices[o     ].position = {l, n, t};
	floor->vertices[o +  1].position = {r, n, t};
	floor->vertices[o +  2].position = {l, f, t};
	floor->vertices[o +  3].position = {r, f, t};
	floor->vertices[o +  4].position = {r, n, t};
	floor->vertices[o +  5].position = {r, f, t};
	floor->vertices[o +  6].position = {r, n, b};
	floor->vertices[o +  7].position = {r, f, b};
	floor->vertices[o +  8].position = {l, n, t};
	floor->vertices[o +  9].position = {l, f, t};
	floor->vertices[o + 10].position = {l, n, b};
	floor->vertices[o + 11].position = {l, f, b};
	floor->vertices[o + 12].position = {l, n, t};
	floor->vertices[o + 13].position = {r, n, t};
	floor->vertices[o + 14].position = {l, n, b};
	floor->vertices[o + 15].position = {r, n, b};
	floor->vertices[o     ].normal   =  vector3_unit_z;
	floor->vertices[o +  1].normal   =  vector3_unit_z;
	floor->vertices[o +  2].normal   =  vector3_unit_z;
	floor->vertices[o +  3].normal   =  vector3_unit_z;
	floor->vertices[o +  4].normal   =  vector3_unit_x;
	floor->vertices[o +  5].normal   =  vector3_unit_x;
	floor->vertices[o +  6].normal   =  vector3_unit_x;
	floor->vertices[o +  7].normal   =  vector3_unit_x;
	floor->vertices[o +  8].normal   = -vector3_unit_x;
	floor->vertices[o +  9].normal   = -vector3_unit_x;
	floor->vertices[o + 10].normal   = -vector3_unit_x;
	floor->vertices[o + 11].normal   = -vector3_unit_x;
	floor->vertices[o + 12].normal   = -vector3_unit_y;
	floor->vertices[o + 13].normal   = -vector3_unit_y;
	floor->vertices[o + 14].normal   = -vector3_unit_y;
	floor->vertices[o + 15].normal   = -vector3_unit_y;
	for(int i = 0; i < 16; ++i)
	{
		float rando = arandom::float_range(0.0f, 1.0f);
		floor->vertices[o + i].colour = {0.0f, 1.0f, rando};
	}
	floor->vertices_count += 16;

	int c = floor->indices_count;
	const int indices_count = 24;
	const int offsets[indices_count] =
	{
		0, 1, 2,
		2, 1, 3,
		4, 6, 5,
		7, 5, 6,
		8, 9, 10,
		11, 10, 9,
		12, 14, 13,
		15, 13, 14,
	};
	for(int i = 0; i < indices_count; ++i)
	{
		floor->indices[c + i] = o + offsets[i];
	}
	floor->indices_count += indices_count;

	return true;
}

// Immediate Mode Functions.....................................................

namespace immediate {

static const int context_max_vertices = 8192;

struct Context
{
	VertexPC vertices[context_max_vertices];
	Matrix4 view_projection;
	GLuint vertex_array;
	GLuint buffer;
	GLuint shader;
	int filled;
};

Context* context;

static void context_create()
{
	context = ALLOCATE(Context, 1);
	Context* c = context;

    glGenVertexArrays(1, &c->vertex_array);
    glBindVertexArray(c->vertex_array);

    glGenBuffers(1, &c->buffer);
    glBindBuffer(GL_ARRAY_BUFFER, c->buffer);

    glBufferData(GL_ARRAY_BUFFER, sizeof(c->vertices), nullptr, GL_DYNAMIC_DRAW);

    GLvoid* offset0 = reinterpret_cast<GLvoid*>(offsetof(VertexPC, position));
    GLvoid* offset1 = reinterpret_cast<GLvoid*>(offsetof(VertexPC, colour));
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPC), offset0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPC), offset1);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

static void context_destroy()
{
	if(context)
	{
		Context* c = context;

		glDeleteBuffers(1, &c->buffer);
		glDeleteVertexArrays(1, &c->vertex_array);
	}
}

static void set_matrices(Matrix4 view, Matrix4 projection)
{
	context->view_projection = projection * view;
}

void draw()
{
	Context* c = context;

    glBindBuffer(GL_ARRAY_BUFFER, c->buffer);
    GLvoid* mapped_buffer = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    if(mapped_buffer)
    {
        memcpy(mapped_buffer, c->vertices, sizeof(VertexPC) * c->filled);
        glUnmapBuffer(GL_ARRAY_BUFFER);
    }

	glUseProgram(context->shader);
	GLint location = glGetUniformLocation(context->shader, "model_view_projection");
	glUniformMatrix4fv(location, 1, GL_TRUE, c->view_projection.elements);
	glBindVertexArray(c->vertex_array);
	glDrawArrays(GL_LINES, 0, c->filled);

	c->filled = 0;
}

void add_line(Vector3 start, Vector3 end, Vector3 colour)
{
	Context* c = context;
	ASSERT(c->filled + 2 < context_max_vertices);
	c->vertices[c->filled] = {start, colour};
	c->vertices[c->filled+1] = {end, colour};
	c->filled += 2;
}

// This is an approximate formula for an ellipse's perimeter. It has the
// greatest error when the ratio of a to b is largest.
static float ellipse_perimeter(float a, float b)
{
	return TAU * sqrt(((a * a) + (b * b)) / 2.0f);
}

static void add_wire_ellipse(Vector3 center, Quaternion orientation, Vector2 radius, Vector3 colour)
{
	Context* c = context;

	const float min_spacing = 0.05f;
	float a = radius.x;
	float b = radius.y;
	int segments = ellipse_perimeter(a, b) / min_spacing;
	ASSERT(c->filled + 2 * segments < context_max_vertices);
	Vector3 point = {a, 0.0f, 0.0f};
	Vector3 position = center + (orientation * point);
	for(int i = 1; i <= segments; ++i)
	{
		Vector3 prior = position;
		float t = (static_cast<float>(i) / segments) * TAU;
		point = {a * cos(t), b * sin(t), 0.0f};
		position = center + (orientation * point);
		add_line(prior, position, colour);
	}
}

static void add_wire_ellipsoid(Vector3 center, Vector3 radius, Vector3 colour)
{
	Quaternion q0 = axis_angle_rotation(vector3_unit_y, +PI_OVER_2);
	Quaternion q1 = axis_angle_rotation(vector3_unit_x, -PI_OVER_2);
	Quaternion q2 = quaternion_identity;
	add_wire_ellipse(center, q0, {radius.z, radius.y}, colour);
	add_wire_ellipse(center, q1, {radius.x, radius.z}, colour);
	add_wire_ellipse(center, q2, {radius.x, radius.y}, colour);
}

void add_wire_aabb(AABB aabb, Vector3 colour)
{
	Vector3 s = aabb.max - aabb.min;

	Vector3 p[8];
	for(int i = 0; i < 8; ++i)
	{
		p[i] = aabb.min;
	}
	p[0] += {0.0f, 0.0f, 0.0f};
	p[1] += {s.x , 0.0f, 0.0f};
	p[2] += {0.0f, s.y , 0.0f};
	p[3] += {s.x , s.y , 0.0f};
	p[4] += {0.0f, 0.0f, s.z };
	p[5] += {s.x , 0.0f, s.z };
	p[6] += {0.0f, s.y , s.z };
	p[7] += {s.x , s.y , s.z };

	immediate::add_line(p[0], p[1], colour);
	immediate::add_line(p[1], p[3], colour);
	immediate::add_line(p[3], p[2], colour);
	immediate::add_line(p[2], p[0], colour);

	immediate::add_line(p[0], p[4], colour);
	immediate::add_line(p[1], p[5], colour);
	immediate::add_line(p[2], p[6], colour);
	immediate::add_line(p[3], p[7], colour);

	immediate::add_line(p[4], p[5], colour);
	immediate::add_line(p[5], p[7], colour);
	immediate::add_line(p[7], p[6], colour);
	immediate::add_line(p[6], p[4], colour);
}

} // namespace immediate

// Debug Drawing Functions......................................................

static void add_aabb_plane(AABB aabb, bih::Flag axis, float clip, Vector3 colour)
{
	Vector3 plane_corner = aabb.min;
	plane_corner[axis] = clip;
	Vector3 p[4];
	for(int i = 0; i < 4; ++i)
	{
		p[i] = plane_corner;
	}
	Vector3 d = aabb.max - aabb.min;
	switch(axis)
	{
		case bih::FLAG_X:
		{
			p[1] += {0.0f, d.y , 0.0f};
			p[2] += {0.0f, 0.0f, d.z };
			p[3] += {0.0f, d.y , d.z };
			break;
		}
		case bih::FLAG_Y:
		{
			p[1] += {d.x , 0.0f, 0.0f};
			p[2] += {0.0f, 0.0f, d.z };
			p[3] += {d.x , 0.0f, d.z };
			break;
		}
		case bih::FLAG_Z:
		{
			p[1] += {d.x , 0.0f, 0.0f};
			p[2] += {0.0f, d.y , 0.0f};
			p[3] += {d.x , d.y , 0.0f};
			break;
		}
		default:
		{
			ASSERT(false);
			break;
		}
	}
	immediate::add_line(p[0], p[1], colour);
	immediate::add_line(p[1], p[3], colour);
	immediate::add_line(p[3], p[2], colour);
	immediate::add_line(p[2], p[0], colour);
	immediate::add_line(p[0], p[3], colour);
}

static void add_bih_node(bih::Node* nodes, bih::Node* node, AABB bounds, int depth, int target_depth)
{
	if(node->flag != bih::FLAG_LEAF)
	{
		bih::Flag axis = node->flag;

		AABB left = bounds;
		left.max[axis] = node->clip[0];
		if(depth == target_depth || target_depth < 0)
		{
			add_aabb_plane(bounds, axis, node->clip[0], {1.0f, 0.0f, 0.0f});
		}
		add_bih_node(nodes, nodes + node->index, left, depth + 1, target_depth);

		AABB right = bounds;
		right.min[axis] = node->clip[1];
		if(depth == target_depth || target_depth < 0)
		{
			add_aabb_plane(bounds, axis, node->clip[1], {0.0f, 0.0f, 1.0f});
		}
		add_bih_node(nodes, nodes + node->index + 1, right, depth + 1, target_depth);
	}
}

static void draw_bih_tree(bih::Tree* tree, int target_depth)
{
	bih::Node* root = tree->nodes;
	AABB bounds = tree->bounds;
	add_bih_node(tree->nodes, root, bounds, 0, target_depth);
	immediate::draw();
}

// Render System Functions......................................................

namespace render {

const char* default_vertex_source = R"(
#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 colour;

uniform mat4x4 model_view_projection;
uniform mat4x4 normal_matrix;

out vec3 surface_normal;
out vec3 surface_colour;

void main()
{
	gl_Position = model_view_projection * vec4(position, 1.0);
	surface_normal = (normal_matrix * vec4(normal, 0.0)).xyz;
	surface_colour = colour;
}
)";

const char* default_fragment_source = R"(
#version 330

layout(location = 0) out vec4 output_colour;

uniform vec3 light_direction;

in vec3 surface_normal;
in vec3 surface_colour;

float half_lambert(vec3 n, vec3 l)
{
	return 0.5 * dot(n, l) + 0.5;
}

float lambert(vec3 n, vec3 l)
{
	return max(dot(n, l), 0.0);
}

void main()
{
	float light = half_lambert(surface_normal, light_direction);
	output_colour = vec4(surface_colour * vec3(light), 1.0);
}
)";

const char* vertex_source_vertex_colour = R"(
#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 colour;

uniform mat4x4 model_view_projection;

out vec3 surface_colour;

void main()
{
	gl_Position = model_view_projection * vec4(position, 1.0);
	surface_colour = colour;
}
)";

const char* fragment_source_vertex_colour = R"(
#version 330

layout(location = 0) out vec4 output_colour;

in vec3 surface_colour;

void main()
{
	output_colour = vec4(surface_colour, 1.0);
}
)";

struct Object
{
	Matrix4 model_view_projection;
	Matrix4 normal_matrix;
	GLuint vertex_array;
	GLuint buffers[2];
	int indices_count;
};

static void object_create(Object* object)
{
	glGenVertexArrays(1, &object->vertex_array);
	glGenBuffers(ARRAY_COUNT(object->buffers), object->buffers);
}

static void object_destroy(Object* object)
{
	glDeleteVertexArrays(1, &object->vertex_array);
	glDeleteBuffers(ARRAY_COUNT(object->buffers), object->buffers);
}

static void object_set_surface(Object* object, VertexPNC* vertices, int vertices_count, u16* indices, int indices_count)
{
	glBindVertexArray(object->vertex_array);

	const int vertex_size = sizeof(VertexPNC);
	GLsizei vertices_size = vertex_size * vertices_count;
	GLvoid* offset1 = reinterpret_cast<GLvoid*>(offsetof(VertexPNC, normal));
	GLvoid* offset2 = reinterpret_cast<GLvoid*>(offsetof(VertexPNC, colour));
	glBindBuffer(GL_ARRAY_BUFFER, object->buffers[0]);
	glBufferData(GL_ARRAY_BUFFER, vertices_size, vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, nullptr);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_size, offset1);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vertex_size, offset2);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	GLsizei indices_size = sizeof(u16) * indices_count;
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, object->buffers[1]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_size, indices, GL_STATIC_DRAW);
	object->indices_count = indices_count;

	glBindVertexArray(0);
}

static void object_set_matrices(Object* object, Matrix4 model, Matrix4 view, Matrix4 projection)
{
	Matrix4 model_view = view * model;
	object->model_view_projection = projection * model_view;
	object->normal_matrix = transpose(inverse_transform(model_view));
}

static void object_generate_floor(Object* object)
{
	Floor floor = {};
	for(int y = 0; y < 10; ++y)
	{
		for(int x = y & 1; x < 10; x += 2)
		{
			Vector3 bottom_left = {0.4f * (x - 10.0f), 0.4f * y, -1.4f};
			Vector3 dimensions = {0.4f, 0.4f, 0.4f};
			bool added = floor_add_box(&floor, bottom_left, dimensions);
			if(!added)
			{
				floor_destroy(&floor);
				return;
			}
		}
	}
	Vector3 wall_position = {1.0f, 0.0f, -1.0f};
	Vector3 wall_dimensions = {0.1f, 2.0f, 1.0f};
	bool added = floor_add_box(&floor, wall_position, wall_dimensions);
	if(!added)
	{
		floor_destroy(&floor);
		return;
	}
	object_set_surface(object, floor.vertices, floor.vertices_count, floor.indices, floor.indices_count);
	floor_destroy(&floor);
}

static void object_generate_player(Object* object)
{
	Floor floor = {};
	Vector3 dimensions = {0.5f, 0.5f, 0.7f};
	Vector3 position = {-dimensions.x / 2.0f, -dimensions.y / 2.0f, -dimensions.z / 2.0f};
	bool added = floor_add_box(&floor, position, dimensions);
	if(!added)
	{
		floor_destroy(&floor);
		return;
	}
	object_set_surface(object, floor.vertices, floor.vertices_count, floor.indices, floor.indices_count);
	floor_destroy(&floor);
}

float hue_to_rgb(float p, float q, float t)
{
	if(t < 0.0f)
	{
		t += 1.0f;
	}
	else if(t > 1.0f)
	{
		t -= 1.0f;
	}
	if(t < (1.0f / 6.0f))
	{
		return p + 6.0f * t * (q - p);
	}
	else if(t < (1.0f / 2.0f))
	{
		return q;
	}
	else if(t < (2.0f / 3.0f))
	{
		return p + 6.0f * ((2.0f / 3.0f) - t) * (q - p);
	}
	return p;
}

Vector3 hsl_to_rgb(Vector3 hsl)
{
	float h = hsl.x;
	float s = hsl.y;
	float l = hsl.z;
	if(s == 0)
	{
		return {l, l, l};
	}
	float q;
	if(l < 0.0f)
	{
		q = l * (1.0f + s);
	}
	else
	{
		q = l + s - (l * s);
	}
	float p = (2.0f * l) - q;
	Vector3 result;
	result.x = hue_to_rgb(p, q, h + (1.0f / 3.0f));
	result.y = hue_to_rgb(p, q, h);
	result.z = hue_to_rgb(p, q, h - (1.0f / 3.0f));
	return result;
}

static void object_generate_terrain(Object* object, Triangle** triangles, int* triangles_count)
{
	const int side = 10;
	const int area = side * side;
	const int columns = side + 1;
	const int vertices_count = columns * columns;

	VertexPNC* vertices = ALLOCATE(VertexPNC, vertices_count);
	if(!vertices)
	{
		return;
	}

	// Generate random heights for each vertex position.
	for(int y = 0; y < columns; ++y)
	{
		for(int x = 0; x < columns; ++x)
		{
			float fx = static_cast<float>(x) - 5.0f;
			float fy = static_cast<float>(y) - 1.0f;
			float fz = arandom::float_range(-0.5f, 0.5f) - 1.0f;
			vertices[y*columns+x].position = {fx, fy, fz};
		}
	}

	// Generate random vertex colours.
	for(int i = 0; i < vertices_count; ++i)
	{
		float h = arandom::float_range(0.0f, 0.1f);
		float s = arandom::float_range(0.7f, 0.9f);
		float l = arandom::float_range(0.5f, 1.0f);
		vertices[i].colour = hsl_to_rgb({h, s, l});
	}

	// Generate the triangle indices.
	int indices_count = 6 * area;
	u16* indices = ALLOCATE(u16, indices_count);
	if(!indices)
	{
		DEALLOCATE(vertices);
		return;
	}

	for(int i = 0; i < area; ++i)
	{
		int o = 6 * i;
		int k = i + i / side;

		indices[o  ] = k + columns + 1;
		indices[o+1] = k + columns;
		indices[o+2] = k;

		indices[o+3] = k + 1;
		indices[o+4] = k + columns + 1;
		indices[o+5] = k;
	}

	// Generate vertex normals from the triangles.
	int* seen = ALLOCATE(int, vertices_count);
	if(!seen)
	{
		DEALLOCATE(vertices);
		DEALLOCATE(indices);
		return;
	}
	for(int i = 0; i < indices_count; i += 3)
	{
		u16 ia = indices[i];
		u16 ib = indices[i+1];
		u16 ic = indices[i+2];
		Vector3 a = vertices[ia].position;
		Vector3 b = vertices[ib].position;
		Vector3 c = vertices[ic].position;
		Vector3 normal = normalise(cross(b - a, c - a));

		u16 v[3] = {ia, ib, ic};
		for(int j = 0; j < 3; ++j)
		{
			GLushort cv = v[j];
			seen[cv] += 1;
			if(seen[cv] == 1)
			{
				vertices[cv].normal = normal;
			}
			else
			{
				// Average each vertex normal with its face normal.
				Vector3 n = lerp(vertices[cv].normal, normal, 1.0f / seen[cv]);
				vertices[cv].normal = normalise(n);
			}
		}
	}
	DEALLOCATE(seen);

	int t_count = 2 * area;
	Triangle* t = ALLOCATE(Triangle, t_count);
	if(!t)
	{
		DEALLOCATE(vertices);
		DEALLOCATE(indices);
		return;
	}
	*triangles = t;
	*triangles_count = t_count;
	for(int i = 0; i < t_count; ++i)
	{
		Vector3 a = vertices[indices[3*i+0]].position;
		Vector3 b = vertices[indices[3*i+1]].position;
		Vector3 c = vertices[indices[3*i+2]].position;
		t[i] = {a, b, c};
	}

	object_set_surface(object, vertices, vertices_count, indices, indices_count);

	DEALLOCATE(vertices);
	DEALLOCATE(indices);
}

// Whole system

GLuint shader;
GLuint shader_vertex_colour;
Matrix4 projection;
int objects_count = 4;
Object objects[4];
Triangle* terrain_triangles;
int terrain_triangles_count;

static bool system_initialise()
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	Object tetrahedron;
	VertexPNC vertices[4] =
	{
		{{+1.0f, +0.0f, -1.0f / sqrt(2.0f)}, { 0.816497f, 0.0f, -0.57735f}, {1.0f, 1.0f, 1.0f}},
		{{-1.0f, +0.0f, -1.0f / sqrt(2.0f)}, {-0.816497f, 0.0f, -0.57735f}, {1.0f, 1.0f, 1.0f}},
		{{+0.0f, +1.0f, +1.0f / sqrt(2.0f)}, { 0.0f, 0.816497f, +0.57735f}, {1.0f, 1.0f, 1.0f}},
		{{+0.0f, -1.0f, +1.0f / sqrt(2.0f)}, { 0.0f,-0.816497f, +0.57735f}, {1.0f, 1.0f, 1.0f}},
	};
	u16 indices[12] =
	{
		0, 1, 2,
		1, 0, 3,
		2, 3, 0,
		3, 2, 1,
	};
	object_create(&tetrahedron);
	object_set_surface(&tetrahedron, vertices, ARRAY_COUNT(vertices), indices, ARRAY_COUNT(indices));
	objects[0] = tetrahedron;

	Object floor;
	object_create(&floor);
	object_generate_floor(&floor);
	objects[1] = floor;

	Object player;
	object_create(&player);
	object_generate_player(&player);
	objects[2] = player;

	Object terrain;
	object_create(&terrain);
	object_generate_terrain(&terrain, &terrain_triangles, &terrain_triangles_count);
	objects[3] = terrain;

	shader = load_shader_program(default_vertex_source, default_fragment_source);
	if(shader == 0)
	{
		LOG_ERROR("The default shader failed to load.");
		return false;
	}

	shader_vertex_colour = load_shader_program(vertex_source_vertex_colour, fragment_source_vertex_colour);
	if(shader_vertex_colour == 0)
	{
		LOG_ERROR("The vertex colour shader failed to load.");
		return false;
	}

	immediate::context_create();
	immediate::context->shader = shader_vertex_colour;

	return true;
}

static void system_terminate(bool functions_loaded)
{
	if(functions_loaded)
	{
		for(int i = 0; i < objects_count; ++i)
		{
			object_destroy(objects + i);
		}
		glDeleteProgram(shader);
		immediate::context_destroy();
	}
}

static void resize_viewport(int width, int height)
{
	const float fov = PI_OVER_2 * (2.0f / 3.0f);
	const float near = 0.05f;
	const float far = 12.0f;
	projection = perspective_projection_matrix(fov, width, height, near, far);
	glViewport(0, 0, width, height);
}

static void system_update(Vector3 position)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Set up the matrices.
	{
		const Vector3 scale = vector3_one;

		Matrix4 model0;
		{
			static float angle = 0.0f;
			angle += 0.02f;

			Vector3 where = {0.0f, 2.0f, 0.0f};
			Quaternion orientation = axis_angle_rotation(vector3_unit_z, angle);
			model0 = compose_transform(where, orientation, scale);
		}

		Matrix4 model1 = matrix4_identity;

		Matrix4 model2;
		{
			Quaternion orientation = quaternion_identity;
			model2 = compose_transform(position, orientation, scale);
		}

		Matrix4 model3 = matrix4_identity;

		const Vector3 camera_position = {0.0f, -3.5f, 1.5f};
		const Vector3 camera_target = {0.0f, 0.0f, 0.5f};
		const Matrix4 view = look_at_matrix(camera_position, camera_target, vector3_unit_z);

		object_set_matrices(objects, model0, view, projection);
		object_set_matrices(objects + 1, model1, view, projection);
		object_set_matrices(objects + 2, model2, view, projection);
		object_set_matrices(objects + 3, model3, view, projection);

		immediate::set_matrices(view, projection);

		glUseProgram(shader);

		Vector3 light_direction = {0.7f, 0.4f, -1.0f};
		light_direction = normalise(-(view * light_direction));
		GLint location = glGetUniformLocation(shader, "light_direction");
		glUniform3fv(location, 1, reinterpret_cast<float*>(&light_direction));
	}

	GLint location0 = glGetUniformLocation(shader, "model_view_projection");
	GLint location1 = glGetUniformLocation(shader, "normal_matrix");

	for(int i = 0; i < objects_count; ++i)
	{
		Object* o = objects + i;
		glUniformMatrix4fv(location0, 1, GL_TRUE, o->model_view_projection.elements);
		glUniformMatrix4fv(location1, 1, GL_TRUE, o->normal_matrix.elements);
		glBindVertexArray(o->vertex_array);
		glDrawElements(GL_TRIANGLES, o->indices_count, GL_UNSIGNED_SHORT, nullptr);
	}

	immediate::add_wire_ellipsoid(position, {0.3f, 0.3f, 0.5f}, {1.0f, 0.0f, 1.0f});
	immediate::draw();
}

} // namespace render

// Speech Function Declarations.................................................

enum class UserKey {Space, Left, Up, Right, Down};

namespace speech_system {

void say_user_key(UserKey key);

} // namespace speech_system

// Main Functions...............................................................

namespace
{
	const char* app_name = "ONE";
	const int window_width = 800;
	const int window_height = 600;
	const double frame_frequency = 1.0 / 60.0;
	const int key_count = 5;

	bool keys_pressed[key_count];
	bool old_keys_pressed[key_count];
	// This counts the frames since the last time the key state changed.
	int edge_counts[key_count];

	Vector3 position;
	World world;
}

static bool key_tapped(UserKey key)
{
	int which = static_cast<int>(key);
	return keys_pressed[which] && edge_counts[which] == 0;
}

static bool key_pressed(UserKey key)
{
	int which = static_cast<int>(key);
	return keys_pressed[which];
}

const char* bool_to_string(bool value)
{
	if(value)
	{
		return "true";
	}
	else
	{
		return "false";
	}
}

static void game_create()
{
	world.triangles = render::terrain_triangles;
	world.triangles_count = render::terrain_triangles_count;
	bool built = bih::build_tree(&world.tree, world.triangles, world.triangles_count);
	ASSERT(built);
}

static void main_update()
{
	// Update input states.
	for(int i = 0; i < key_count; ++i)
	{
		if(keys_pressed[i] != old_keys_pressed[i])
		{
			edge_counts[i] = 0;
			old_keys_pressed[i] = keys_pressed[i];
		}
		else
		{
			edge_counts[i] += 1;
		}
	}

	// Update the player's movement state with the input.
	struct {float x, y;} d = {0.0f, 0.0f};
	if(key_pressed(UserKey::Left))
	{
		d.x -= 1.0f;
	}
	if(key_pressed(UserKey::Right))
	{
		d.x += 1.0f;
	}
	if(key_pressed(UserKey::Up))
	{
		d.y += 1.0f;
	}
	if(key_pressed(UserKey::Down))
	{
		d.y -= 1.0f;
	}
	float l = sqrt((d.x * d.x) + (d.y * d.y));
	Vector3 velocity = vector3_zero;
	if(l != 0.0f)
	{
		d.x /= l;
		d.y /= l;
		const float speed = 0.08f;
		velocity.x += speed * d.x;
		velocity.y += speed * d.y;
	}

	// Say any inputs the player is currently pressing.
	if(key_tapped(UserKey::Left))  speech_system::say_user_key(UserKey::Left);
	if(key_tapped(UserKey::Up))    speech_system::say_user_key(UserKey::Up);
	if(key_tapped(UserKey::Right)) speech_system::say_user_key(UserKey::Right);
	if(key_tapped(UserKey::Down))  speech_system::say_user_key(UserKey::Down);
	if(key_tapped(UserKey::Space)) speech_system::say_user_key(UserKey::Space);

	Vector3 radius = {0.3f, 0.3f, 0.5f};
	position = collide_and_slide(position, radius, velocity, -0.096f * vector3_unit_z, &world);

	// Respawn if below the out-of-bounds plane.
	if(position.z < -4.0f)
	{
		position = vector3_zero;
	}

	render::system_update(position);
}

// OpenGL Function Loading......................................................

#if defined(OS_LINUX)
#include <GL/glx.h>
#define GET_PROC(name) \
	(*glXGetProcAddress)(reinterpret_cast<const GLubyte*>(name))

#elif defined(OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN 1
#define NOMINMAX
#include <Windows.h>
#if defined(near)
#undef near
#endif
#if defined(far)
#undef far
#endif

namespace
{
    HMODULE gl_module;
}

static PROC windows_get_proc_address(const char* name)
{
    PROC address = wglGetProcAddress(reinterpret_cast<LPCSTR>(name));
    if(address)
    {
        return address;
    }
    if(!gl_module)
    {
        gl_module = GetModuleHandleA("OpenGL32.dll");
    }
    return GetProcAddress(gl_module, reinterpret_cast<LPCSTR>(name));
}

#define GET_PROC(name) (*windows_get_proc_address)(name)
#endif

static bool ogl_load_functions()
{
	p_glClear = reinterpret_cast<void (APIENTRYA*)(GLbitfield)>(GET_PROC("glClear"));
	p_glEnable = reinterpret_cast<void (APIENTRYA*)(GLenum)>(GET_PROC("glEnable"));
	p_glViewport = reinterpret_cast<void (APIENTRYA*)(GLint, GLint, GLsizei, GLsizei)>(GET_PROC("glViewport"));

	p_glDrawArrays = reinterpret_cast<void (APIENTRYA*)(GLenum, GLint, GLsizei)>(GET_PROC("glDrawArrays"));
	p_glDrawElements = reinterpret_cast<void (APIENTRYA*)(GLenum, GLsizei, GLenum, const void*)>(GET_PROC("glDrawElements"));

	p_glBindVertexArray = reinterpret_cast<void (APIENTRYA*)(GLuint)>(GET_PROC("glBindVertexArray"));
	p_glDeleteVertexArrays = reinterpret_cast<void (APIENTRYA*)(GLsizei, const GLuint*)>(GET_PROC("glDeleteVertexArrays"));
	p_glGenVertexArrays = reinterpret_cast<void (APIENTRYA*)(GLsizei, GLuint*)>(GET_PROC("glGenVertexArrays"));

	p_glBindBuffer = reinterpret_cast<void (APIENTRYA*)(GLenum, GLuint)>(GET_PROC("glBindBuffer"));
	p_glBufferData = reinterpret_cast<void (APIENTRYA*)(GLenum, GLsizeiptr, const void*, GLenum)>(GET_PROC("glBufferData"));
	p_glDeleteBuffers = reinterpret_cast<void (APIENTRYA*)(GLsizei, const GLuint*)>(GET_PROC("glDeleteBuffers"));
	p_glGenBuffers = reinterpret_cast<void (APIENTRYA*)(GLsizei, GLuint*)>(GET_PROC("glGenBuffers"));
	p_glMapBuffer = reinterpret_cast<void* (APIENTRYA*)(GLenum, GLenum)>(GET_PROC("glMapBuffer"));
	p_glUnmapBuffer = reinterpret_cast<GLboolean (APIENTRYA*)(GLenum)>(GET_PROC("glUnmapBuffer"));

	p_glAttachShader = reinterpret_cast<void (APIENTRYA*)(GLuint, GLuint)>(GET_PROC("glAttachShader"));
	p_glCompileShader = reinterpret_cast<void (APIENTRYA*)(GLuint)>(GET_PROC("glCompileShader"));
	p_glCreateProgram = reinterpret_cast<GLuint (APIENTRYA*)(void)>(GET_PROC("glCreateProgram"));
	p_glCreateShader = reinterpret_cast<GLuint (APIENTRYA*)(GLenum)>(GET_PROC("glCreateShader"));
	p_glDeleteProgram = reinterpret_cast<void (APIENTRYA*)(GLuint)>(GET_PROC("glDeleteProgram"));
	p_glDeleteShader = reinterpret_cast<void (APIENTRYA*)(GLuint)>(GET_PROC("glDeleteShader"));
	p_glDetachShader = reinterpret_cast<void (APIENTRYA*)(GLuint, GLuint)>(GET_PROC("glDetachShader"));
	p_glEnableVertexAttribArray = reinterpret_cast<void (APIENTRYA*)(GLuint)>(GET_PROC("glEnableVertexAttribArray"));
	p_glGetProgramInfoLog = reinterpret_cast<void (APIENTRYA*)(GLuint, GLsizei, GLsizei*, GLchar*)>(GET_PROC("glGetProgramInfoLog"));
	p_glGetProgramiv = reinterpret_cast<void (APIENTRYA*)(GLuint, GLenum, GLint*)>(GET_PROC("glGetProgramiv"));
	p_glGetShaderInfoLog = reinterpret_cast<void (APIENTRYA*)(GLuint, GLsizei, GLsizei*, GLchar*)>(GET_PROC("glGetShaderInfoLog"));
	p_glGetShaderiv = reinterpret_cast<void (APIENTRYA*)(GLuint, GLenum, GLint*)>(GET_PROC("glGetShaderiv"));
	p_glGetUniformLocation = reinterpret_cast<GLint (APIENTRYA*)(GLuint, const GLchar*)>(GET_PROC("glGetUniformLocation"));
	p_glLinkProgram = reinterpret_cast<void (APIENTRYA*)(GLuint)>(GET_PROC("glLinkProgram"));
	p_glShaderSource = reinterpret_cast<void (APIENTRYA*)(GLuint, GLsizei, const GLchar* const*, const GLint*)>(GET_PROC("glShaderSource"));
	p_glUniform3fv = reinterpret_cast<void (APIENTRYA*)(GLint, GLsizei, const GLfloat*)>(GET_PROC("glUniform3fv"));
	p_glUniformMatrix4fv = reinterpret_cast<void (APIENTRYA*)(GLint, GLsizei, GLboolean, const GLfloat*)>(GET_PROC("glUniformMatrix4fv"));
	p_glUseProgram = reinterpret_cast<void (APIENTRYA*)(GLuint)>(GET_PROC("glUseProgram"));
	p_glVertexAttribPointer = reinterpret_cast<void (APIENTRYA*)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*)>(GET_PROC("glVertexAttribPointer"));

	int failure_count = 0;

	failure_count += p_glClear == nullptr;
	failure_count += p_glEnable == nullptr;
	failure_count += p_glViewport == nullptr;

	failure_count += p_glDrawArrays == nullptr;
	failure_count += p_glDrawElements == nullptr;

	failure_count += p_glBindVertexArray == nullptr;
	failure_count += p_glDeleteVertexArrays == nullptr;
	failure_count += p_glGenVertexArrays == nullptr;

	failure_count += p_glBindBuffer == nullptr;
	failure_count += p_glBufferData == nullptr;
	failure_count += p_glDeleteBuffers == nullptr;
	failure_count += p_glGenBuffers == nullptr;
	failure_count += p_glMapBuffer == nullptr;
	failure_count += p_glUnmapBuffer == nullptr;

	failure_count += p_glAttachShader == nullptr;
	failure_count += p_glCompileShader == nullptr;
	failure_count += p_glCreateProgram == nullptr;
	failure_count += p_glCreateShader == nullptr;
	failure_count += p_glDeleteProgram == nullptr;
	failure_count += p_glDeleteShader == nullptr;
	failure_count += p_glDetachShader == nullptr;
	failure_count += p_glEnableVertexAttribArray == nullptr;
	failure_count += p_glGetProgramInfoLog == nullptr;
	failure_count += p_glGetProgramiv == nullptr;
	failure_count += p_glGetShaderInfoLog == nullptr;
	failure_count += p_glGetShaderiv == nullptr;
	failure_count += p_glGetUniformLocation == nullptr;
	failure_count += p_glLinkProgram == nullptr;
	failure_count += p_glShaderSource == nullptr;
	failure_count += p_glUniform3fv == nullptr;
	failure_count += p_glUniformMatrix4fv == nullptr;
	failure_count += p_glUseProgram == nullptr;
	failure_count += p_glVertexAttribPointer == nullptr;

	return failure_count == 0;
}

// Platform-Specific Implementations============================================

#if defined(OS_LINUX)

// Logging Functions............................................................

void log_add_message(LogLevel level, const char* format, ...)
{
	va_list arguments;
	va_start(arguments, format);
	FILE* stream;
	switch(level)
	{
		case LogLevel::Error:
			stream = stderr;
			break;
		case LogLevel::Debug:
			stream = stdout;
			break;
	}
	vfprintf(stream, format, arguments);
	va_end(arguments);
	fputc('\n', stream);
}

// Clock Functions..............................................................

#include <ctime>

void initialise_clock(Clock* clock)
{
	struct timespec resolution;
	clock_getres(CLOCK_MONOTONIC, &resolution);
	s64 nanoseconds = resolution.tv_nsec + resolution.tv_sec * 1e9;
	clock->frequency = static_cast<double>(nanoseconds) / 1.0e9;
}

double get_time(Clock* clock)
{
	struct timespec timestamp;
	clock_gettime(CLOCK_MONOTONIC, &timestamp);
	s64 nanoseconds = timestamp.tv_nsec + timestamp.tv_sec * 1e9;
	return static_cast<double>(nanoseconds) * clock->frequency;
}

void go_to_sleep(Clock* clock, double amount_to_sleep)
{
	struct timespec requested_time;
	requested_time.tv_sec = 0;
	requested_time.tv_nsec = static_cast<s64>(1.0e9 * amount_to_sleep);
	clock_nanosleep(CLOCK_MONOTONIC, 0, &requested_time, nullptr);
}

// Speech Functions.............................................................

#include <libspeechd.h>
#include <unistd.h>

namespace speech_system {

namespace
{
	SPDConnection* connection;
}

static bool initialise()
{
	const char* connection_name = "main";
	char* username = getlogin();
	connection = spd_open(app_name, connection_name, username, SPD_MODE_SINGLE);
	if(!connection)
	{
		LOG_ERROR("Speech Dispatcher failed initialisation.");
		return false;
	}

	return true;
}

static void terminate()
{
	if(connection)
	{
		spd_close(connection);
	}
}

static const char* get_user_key_description(UserKey key)
{
	switch(key)
	{
		case UserKey::Left:  return "left";
		case UserKey::Up:    return "up";
		case UserKey::Right: return "right";
		case UserKey::Down:  return "down";
		case UserKey::Space: return "space";
	}
	return "";
}

void say_user_key(UserKey key)
{
	const char* text = get_user_key_description(key);
	spd_say(connection, SPD_TEXT, text);
}

} // namespace speech_system

// Platform Main Functions......................................................

#include <X11/X.h>
#include <X11/Xlib.h>

#include <cstdlib>

namespace
{
	Display* display;
	XVisualInfo* visual_info;
	Colormap colormap;
	Window window;
	Atom wm_delete_window;
	GLXContext rendering_context;
	bool functions_loaded;
}

static bool main_create()
{
	// Connect to the X server, which is used for display and input services.
	display = XOpenDisplay(nullptr);
	if(!display)
	{
		LOG_ERROR("X Display failed to open.");
		return false;
	}

	// Choose the abstract "Visual" type that will be used to describe both the
	// window and the OpenGL rendering context.
	GLint visual_attributes[] = {GLX_RGBA, GLX_DEPTH_SIZE, 24,GLX_DOUBLEBUFFER, None};
	visual_info = glXChooseVisual(display, DefaultScreen(display), visual_attributes);
	if(!visual_info)
	{
		LOG_ERROR("Wasn't able to choose an appropriate Visual type given the requested attributes. [The Visual type contains information on color mappings for the display hardware]");
		return false;
	}

	// Create the Window.
	{
		int screen = DefaultScreen(display);
		Window root = RootWindow(display, screen);
		Visual* visual = visual_info->visual;
		colormap = XCreateColormap(display, root, visual, AllocNone);

		int width = window_width;
		int height = window_height;
		int depth = visual_info->depth;

		XSetWindowAttributes attributes = {};
		attributes.colormap = colormap;
		attributes.event_mask = StructureNotifyMask;
		unsigned long mask = CWColormap | CWEventMask;
		window = XCreateWindow(display, root, 0, 0, width, height, 0, depth, InputOutput, visual, mask, &attributes);
	}

	// Register to receive window close messages.
	wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);
	XSetWMProtocols(display, window, &wm_delete_window, 1);

	XStoreName(display, window, app_name);
	XSetIconName(display, window, app_name);

	// Create the rendering context for OpenGL. The rendering context can only
	// be "made current" after the window is mapped (with XMapWindow).
	rendering_context = glXCreateContext(display, visual_info, nullptr, True);
	if(!rendering_context)
	{
		LOG_ERROR("Couldn't create a GLX rendering context.");
		return false;
	}

	XMapWindow(display, window);

	Bool made_current = glXMakeCurrent(display, window, rendering_context);
	if(!made_current)
	{
		LOG_ERROR("Failed to attach the GLX context to the window.");
		return false;
	}

	arandom::seed(time(nullptr));

	functions_loaded = ogl_load_functions();
	if(!functions_loaded)
	{
		LOG_ERROR("OpenGL functions could not be loaded!");
		return false;
	}
	bool initialised = render::system_initialise();
	if(!initialised)
	{
		LOG_ERROR("Render system failed initialisation.");
		return false;
	}
	render::resize_viewport(window_width, window_height);

	initialised = speech_system::initialise();
	if(!initialised)
	{
		LOG_ERROR("Speech system failed initialisation.");
		return false;
	}

	game_create();

	return true;
}

static void main_destroy()
{
	speech_system::terminate();
	render::system_terminate(functions_loaded);

	if(visual_info)
	{
		XFree(visual_info);
	}
	if(display)
	{
		if(rendering_context)
		{
			glXDestroyContext(display, rendering_context);
		}
		if(colormap != None)
		{
			XFreeColormap(display, colormap);
		}
		XCloseDisplay(display);
	}
}

static void main_loop()
{
	Clock frame_clock;
	initialise_clock(&frame_clock);
	for(;;)
	{
		// Record when the frame begins.
		double frame_start_time = get_time(&frame_clock);

		main_update();
		glXSwapBuffers(display, window);

		// Handle window events.
		while(XPending(display) > 0)
		{
			XEvent event = {};
			XNextEvent(display, &event);
			switch(event.type)
			{
				case ClientMessage:
				{
					XClientMessageEvent client_message = event.xclient;
					Atom message = static_cast<Atom>(client_message.data.l[0]);
					if(message == wm_delete_window)
					{
						XDestroyWindow(display, window);
						return;
					}
					break;
				}
				case ConfigureNotify:
				{
					XConfigureRequestEvent configure = event.xconfigurerequest;
					render::resize_viewport(configure.width, configure.height);
					break;
				}
			}
		}

		// Get key states for input.
		char keys[32];
		XQueryKeymap(display, keys);
		int keysyms[key_count] = {XK_space, XK_Left, XK_Up, XK_Right, XK_Down};
		for(int i = 0; i < key_count; ++i)
		{
			int code = XKeysymToKeycode(display, keysyms[i]);
			keys_pressed[i] = keys[code / 8] & (1 << (code % 8));
		}

		// Sleep off any remaining time until the next frame.
		double frame_thusfar = get_time(&frame_clock) - frame_start_time;
		if(frame_thusfar < frame_frequency)
		{
			go_to_sleep(&frame_clock, frame_frequency - frame_thusfar);
		}
	}
}

int main(int argc, char** argv)
{
	static_cast<void>(argc);
	static_cast<void>(argv);

	if(!main_create())
	{
		main_destroy();
		return EXIT_FAILURE;
	}
	main_loop();
	main_destroy();
	return 0;
}

#elif defined(OS_WINDOWS)

// Logging Functions............................................................

void log_add_message(LogLevel level, const char* format, ...)
{
	va_list arguments;
	va_start(arguments, format);

	const int buffer_capacity = 128;
	char buffer[buffer_capacity];
	int written = vsnprintf(buffer, buffer_capacity - 1, format, arguments);
	va_end(arguments);

	if(written < 0)
	{
		written = buffer_capacity - 2;
	}
	if(written < buffer_capacity - 1)
	{
		buffer[written] = '\n';
		buffer[written + 1] = '\0';
		written += 2;
	}

	OutputDebugStringA(buffer);
}

// Clock Functions..............................................................

void initialise_clock(Clock* clock)
{
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	clock->frequency = 1.0 / frequency.QuadPart;
}

double get_time(Clock* clock)
{
	LARGE_INTEGER count;
	QueryPerformanceCounter(&count);
	return count.QuadPart * clock->frequency;
}

void go_to_sleep(Clock* clock, double amount_to_sleep)
{
    int milliseconds = 1000 * amount_to_sleep;
    Sleep(milliseconds);
}

// Platform Main Functions......................................................

#include <ctime>

namespace
{
    HWND window;
    HDC device_context;
    HGLRC rendering_context;
    bool ogl_functions_loaded;
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT message, WPARAM w_param, LPARAM l_param)
{
    switch(message)
    {
        case WM_CLOSE:
        {
            PostQuitMessage(0);
            return 0;
        }
        case WM_DESTROY:
        {
            HGLRC rc = wglGetCurrentContext();
            if(rc)
            {
                HDC dc = wglGetCurrentDC();
                wglMakeCurrent(nullptr, nullptr);
                ReleaseDC(hwnd, dc);
                wglDeleteContext(rc);
            }
            DestroyWindow(hwnd);
            if(hwnd == window)
            {
                window = nullptr;
            }
            return 0;
        }
        default:
        {
            return DefWindowProc(hwnd, message, w_param, l_param);
        }
    }
}

static bool main_create(HINSTANCE instance, int show_command)
{
    WNDCLASSEXA window_class = {};
    window_class.cbSize = sizeof window_class;
    window_class.style = CS_HREDRAW | CS_VREDRAW;
    window_class.lpfnWndProc = WindowProc;
    window_class.hInstance = instance;
    window_class.hIcon = LoadIcon(instance, IDI_APPLICATION);
    window_class.hIconSm = static_cast<HICON>(LoadIcon(instance, IDI_APPLICATION));
    window_class.hCursor = LoadCursor(nullptr, IDC_ARROW);
    window_class.lpszClassName = "OneWindowClass";
    ATOM registered_class = RegisterClassExA(&window_class);
    if(registered_class == 0)
    {
        LOG_ERROR("Failed to register the window class.");
        return false;
    }

    DWORD window_style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    window = CreateWindowExA(WS_EX_APPWINDOW, MAKEINTATOM(registered_class), window_title, window_style, CW_USEDEFAULT, CW_USEDEFAULT, window_width, window_height, nullptr, nullptr, instance, nullptr);
    if(!window)
    {
        LOG_ERROR("Failed to create the window.");
        return false;
    }

    device_context = GetDC(window);
    if(!device_context)
    {
        LOG_ERROR("Couldn't obtain the device context.");
        return false;
    }

    PIXELFORMATDESCRIPTOR descriptor = {};
    descriptor.nSize = sizeof descriptor;
    descriptor.nVersion = 1;
    descriptor.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER | PFD_DEPTH_DONTCARE;
    descriptor.iPixelType = PFD_TYPE_RGBA;
    descriptor.cColorBits = 32;
    descriptor.iLayerType = PFD_MAIN_PLANE;
    int format_index = ChoosePixelFormat(device_context, &descriptor);
    if(format_index == 0)
    {
        LOG_ERROR("Failed to set up the pixel format.");
        return false;
    }
    if(SetPixelFormat(device_context, format_index, &descriptor) == FALSE)
    {
        LOG_ERROR("Failed to set up the pixel format.");
        return false;
    }

    rendering_context = wglCreateContext(device_context);
    if(!rendering_context)
    {
        LOG_ERROR("Couldn't create the rendering context.");
        return false;
    }

    ShowWindow(window, show_command);

    // Set it to be this thread's rendering context.
    if(wglMakeCurrent(device_context, rendering_context) == FALSE)
    {
        LOG_ERROR("Couldn't set this thread's rendering context "
            "(wglMakeCurrent failed).");
        return false;
    }

    arandom::seed(time(nullptr));

    ogl_functions_loaded = ogl_load_functions();
    if(!ogl_functions_loaded)
    {
        LOG_ERROR("OpenGL functions could not be loaded!");
        return false;
    }
    bool initialised = render_system::initialise();
    if(!initialised)
    {
        LOG_ERROR("Render system failed initialisation.");
        return false;
    }
    render_system::resize_viewport(window_width, window_height);

    return true;
}

static void main_destroy()
{
    render_system::terminate(ogl_functions_loaded);

    if(rendering_context)
    {
        wglMakeCurrent(nullptr, nullptr);
        ReleaseDC(window, device_context);
        wglDeleteContext(rendering_context);
    }
    else if(device_context)
    {
        ReleaseDC(window, device_context);
    }
    if(window)
    {
        DestroyWindow(window);
    }
}

static int main_loop()
{
	Clock frame_clock = {};
	initialise_clock(&frame_clock);
	MSG msg = {};
	for(;;)
	{
		// Record when the frame begins.
		double frame_start_time = get_time(&frame_clock);

		//
		main_update();
		SwapBuffers(device_context);

		// Handle all window messages queued up during the past frame.
		while(PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
		{
			if(msg.message == WM_QUIT)
			{
				return msg.wParam;
			}
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}

		// Get key states for input.
		const int virtual_keys[key_count] = {VK_SPACE, VK_LEFT, VK_UP, VK_RIGHT, VK_DOWN};
		for(int i = 0; i < key_count; ++i)
		{
			keys_pressed[i] = 0x8000 & GetKeyState(virtual_keys[i]);
		}

		// Sleep off any remaining time until the next frame.
		double frame_thusfar = get_time(&frame_clock) - frame_start_time;
		if(frame_thusfar < frame_frequency)
		{
			go_to_sleep(&frame_clock, frame_frequency - frame_thusfar);
		}
	}
	return 0;
}

int CALLBACK WinMain(HINSTANCE instance, HINSTANCE previous_instance, LPSTR command_line, int show_command)
{
	static_cast<void>(previous_instance);
	static_cast<void>(command_line);

	bool created = main_create(instance, show_command);
	if(!created)
	{
		main_destroy();
		return 0;
	}
	return main_loop();
}

#endif // defined(OS_WINDOWS)
