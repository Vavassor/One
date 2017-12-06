/* Written in 2017 by Andrew Dawson

To the extent possible under law, the author(s) have dedicated all copyright and
related and neighboring rights to this software to the public domain worldwide.
This software is distributed without any warranty.

You should have received a copy of the CC0 Public Domain Dedication along with
this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.


Build Instructions..............................................................

For playing, use these commands:

-for Windows, using the visual studio compiler
cl /O2 /DNDEBUG main.cpp kernel32.lib user32.lib gdi32.lib opengl32.lib ole32.lib avrt.lib /link /out:One.exe

-for Linux
g++ -o One -std=c++0x -O3 -DNDEBUG main.cpp -lGL -lX11 -lpthread -lasound

For debugging, use these commands:

-for Windows, using the visual studio compiler
cl /Od /Wall main.cpp kernel32.lib user32.lib gdi32.lib opengl32.lib ole32.lib avrt.lib /link /debug /out:One.exe

-for Linux
g++ -o One -std=c++0x -O0 -g3 -Wall -fmessage-length=0 main.cpp -lGL -lX11 -lpthread -lasound


Table Of Contents...............................................................

1. Utility and Math
	§1.1 Globally-Useful Things
	§1.2 Strings
	§1.3 Sorting
	§1.4 Clock Declarations
	§1.5 Logging Declarations
	§1.6 Atomic Declarations
	§1.7 Profile Declarations
	§1.8 Immediate Mode Drawing Declarations
	§1.9 Random Number Generation
	§1.10 Vectors
	§1.11 Quaternions
	§1.12 Matrices

2. Physics
	§2.1 Geometry Functions
	§2.2 Bounding Volume Hierarchy
	§2.3 Bounding Interval Hierarchy
	§2.4 Collision Functions
	§2.5 Frustum Functions

3. Video
	§3.1 OpenGL Function and Type Declarations
	§3.2 Shader
	§3.3 Vertex Types and Packing
	§3.4 Floor
	§3.5 Immediate Mode Drawing
	§3.6 Debug Visualisers
	§3.7 Colour
	§3.8 Text
	§3.9 Input
	§3.10 Scroll Panel
	§3.11 Tweaker
	§3.12 Oscilloscope
	§3.13 Profile Inspector
	§3.14 Render System
		§3.14.1 Shader Sources

4. Audio
	§4.1 Format Conversion
	§4.2 Oscillators
	§4.3 Envelopes
	§4.4 Filters
	§4.5 Effects
	§4.6 Voice
	§4.7 Voice Map
	§4.8 Track
	§4.9 Stream
	§4.10 Message Queue
	§4.11 Generate Oscillation
	§4.12 Audio System Declarations

5. Game
	§5.1 Game
	§5.2 Audio

6. Profile
	§6.1 Spin Lock
	§6.2 Caller
	§6.3 Global Profile

7. OpenGL Function Loading

8. Compiler-Specific Implementations
	§8.1 Atomic Functions
	§8.2 Timing

9. Platform-Specific Implementations
	§9.1 Logging Functions
	§9.2 Clock Functions
	§9.3 Audio Functions
		§9.3.1 Device
		§9.3.2 System Functions
	§9.4 Platform Main Functions
*/

#if defined(__linux__)
#define OS_LINUX
#elif defined(_WIN32)
#define OS_WINDOWS
#else
#error Failed to figure out what operating system this is.
#endif

#if defined(_MSC_VER)
#define COMPILER_MSVC
#elif defined(__GNUC__)
#define COMPILER_GCC
#else
#error Failed to figure out the compiler used.
#endif

#if defined(__i386__) || defined(_M_IX86)
#define INSTRUCTION_SET_X86
#elif defined(__amd64__) || defined(_M_X64)
#define INSTRUCTION_SET_X64
#elif defined(__arm__) || defined(_M_ARM)
#define INSTRUCTION_SET_ARM
#else
#error Failed to figure out what instruction set the CPU on this computer uses.
#endif

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <cmath>

using std::fmin;
using std::fmax;
using std::abs;
using std::sqrt;
using std::isfinite;
using std::signbit;
using std::sin;
using std::cos;

// §1.1 Globally-Useful Things..................................................

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

#if defined(COMPILER_MSVC)
#define RESTRICT __restrict
#elif defined(COMPILER_GCC)
#define RESTRICT __restrict__
#endif

#define ASSERT(expression)\
	assert(expression)

#define ALLOCATE(type, count)\
	static_cast<type*>(calloc((count), sizeof(type)))
#define REALLOCATE(memory, type, count)\
	static_cast<type*>(realloc((memory), sizeof(type) * (count)))
#define DEALLOCATE(memory)\
	free(memory)
#define SAFE_DEALLOCATE(memory)\
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

#define ENSURE_ARRAY_SIZE(array, extra)\
	ensure_array_size(reinterpret_cast<void**>(&array), &array##_capacity, sizeof(*(array)), array##_count + (extra))

#define ARRAY_COUNT(array)\
	static_cast<int>(sizeof(array) / sizeof(*(array)))

// Prefer these for integers and fmax() and fmin() for floating-point numbers.
#define MAX(a, b)\
	(((a) > (b)) ? (a) : (b))
#define MIN(a, b)\
	(((a) < (b)) ? (a) : (b))

static float clamp(float a, float min, float max)
{
	return fmin(fmax(a, min), max);
}

float lerp(float v0, float v1, float t)
{
	return (1.0f - t) * v0 + t * v1;
}

float unlerp(float v0, float v1, float t)
{
	ASSERT(v0 != v1);
	return (t - v0) / (v1 - v0);
}

static bool is_power_of_two(unsigned int x)
{
	return (x != 0) && !(x & (x - 1));
}

static u32 next_power_of_two(u32 x)
{
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x + 1;
}

static bool can_use_bitwise_and_to_cycle(int count)
{
	return is_power_of_two(count);
}

static int mod(int x, int n)
{
	return (x % n + n) % n;
}

static bool almost_zero(float x)
{
	return x > -1e-6f && x < 1e-6f;
}

static bool almost_one(float x)
{
	return abs(x - 1.0f) <= 0.5e-5f;
}

static bool almost_equals(float x, float y)
{
	return abs(x - y) < 1e-6f;
}

const float tau = 6.28318530717958647692f;
const float pi = 3.14159265358979323846f;
const float pi_over_2 = 1.57079632679489661923f;

// §1.2 Strings.................................................................

static int string_size(const char* string)
{
	ASSERT(string);
	const char* s;
	for(s = string; *s; ++s);
	return s - string;
}

static const char* bool_to_string(bool b)
{
	if(b)
	{
		return "true";
	}
	else
	{
		return "false";
	}
}

// Use this instead of the function strncpy from string.h, because strncpy
// doesn't guarantee null-termination and should be considered hazardous.
static int copy_string(char* RESTRICT to, int to_size, const char* RESTRICT from)
{
	ASSERT(from);
	ASSERT(to);
	int i;
	for(i = 0; i < to_size - 1; ++i)
	{
		if(from[i] == '\0')
		{
			break;
		}
		to[i] = from[i];
	}
	to[i] = '\0';
	ASSERT(i < to_size);
	return i;
}

static void empty_string(char* s)
{
	s[0] = '\0';
}

static int find_char(const char* s, char c)
{
	int i;
	for(i = 0; *s != c; ++s, ++i)
	{
		if(!*s)
		{
			return -1;
		}
	}
	return i;
}

static int find_last_char(const char* s, char c, int limit)
{
	int result = -1;
	int i = 0;
	do
	{
		if(*s == c)
		{
			result = i;
		}
	} while(*s++ && i++ < limit);
	return result;
}

// §1.3 Sorting.................................................................

#define DEFINE_INSERTION_SORT(type, after, suffix)\
	static void insertion_sort_##suffix(type* a, int count)\
	{\
		for(int i = 1; i < count; ++i)\
		{\
			type x = a[i];\
			int j = i - 1;\
			for(; j >= 0 && after(a[j], x); --j)\
			{\
				a[j + 1] = a[j];\
			}\
			a[j + 1] = x;\
		}\
	}

#define DEFINE_QUICK_SORT(type, after, suffix)\
	static void quick_sort_innards(type* a, int left, int right)\
	{\
		while(left + 16 < right)\
		{\
			int middle = (left + right) / 2;\
			type median;\
			if(after(a[left], a[right]))\
			{\
				if(after(a[middle], a[left]))\
				{\
					median = a[left];\
				}\
				else if(after(a[middle], a[right]))\
				{\
					median = a[middle];\
				}\
				else\
				{\
					median = a[right];\
				}\
			}\
			else\
			{\
				if(after(a[middle], a[right]))\
				{\
					median = a[right];\
				}\
				else if(after(a[middle], a[left]))\
				{\
					median = a[middle];\
				}\
				else\
				{\
					median = a[left];\
				}\
			}\
			int i = left - 1;\
			int j = right + 1;\
			int pivot;\
			for(;;)\
			{\
				do {j -= 1;} while(after(median, a[j]));\
				do {i += 1;} while(after(a[i], median));\
				if(i >= j)\
				{\
					pivot = j;\
					break;\
				}\
				else\
				{\
					type temp = a[i];\
					a[i] = a[j];\
					a[j] = temp;\
				}\
			}\
			quick_sort_innards(a, left, pivot);\
			left = pivot + 1;\
		}\
	}\
	\
	DEFINE_INSERTION_SORT(type, after, suffix);\
	\
	static void quick_sort_##suffix(type* a, int count)\
	{\
		quick_sort_innards(a, 0, count - 1);\
		insertion_sort(a, count);\
	}

// §1.4 Clock Declarations......................................................

struct Clock
{
	double frequency;
};

void initialise_clock(Clock* clock);
double get_time(Clock* clock);
void go_to_sleep(Clock* clock, double amount_to_sleep);

u64 get_timestamp_from_system();

void yield();
u64 get_timestamp();

// §1.5 Logging Declarations....................................................

enum class LogLevel
{
	Debug,
	Error,
};

void log_add_message(LogLevel level, const char* format, ...);

#define LOG_ERROR(format, ...)\
	log_add_message(LogLevel::Error, format, ##__VA_ARGS__)

#if defined(NDEBUG)
#define LOG_DEBUG(format, ...) // do nothing
#else
#define LOG_DEBUG(format, ...)\
	log_add_message(LogLevel::Debug, format, ##__VA_ARGS__)
#endif

// §1.6 Atomic Declarations.....................................................

#if defined(COMPILER_MSVC)
typedef long AtomicFlag;
#elif defined(COMPILER_GCC)
typedef bool AtomicFlag;
#endif

typedef long AtomicInt;

bool atomic_flag_test_and_set(AtomicFlag* flag);
void atomic_flag_clear(AtomicFlag* flag);

void atomic_int_store(AtomicInt* i, long value);
long atomic_int_load(AtomicInt* i);
long atomic_int_add(AtomicInt* augend, long addend);
long atomic_int_subtract(AtomicInt* minuend, long subtrahend);

bool atomic_compare_exchange(volatile u32* p, u32 expected, u32 desired);

// §1.7 Profile Declarations....................................................

#if !defined(NDEBUG)
#define PROFILE_ENABLED
#endif

#define PROFILE_MACRO_PASTE2(a, b) a##b
#define PROFILE_MACRO_PASTE(a, b)\
	PROFILE_MACRO_PASTE2(a, b)

#if defined(COMPILER_MSVC)
#define PROFILE_FUNCTION_NAME __FUNCSIG__
#elif defined(COMPILER_GCC)
#define PROFILE_FUNCTION_NAME __PRETTY_FUNCTION__
#endif

#if defined(PROFILE_ENABLED)

#define PROFILE_BEGIN()\
	profile::begin_period(PROFILE_FUNCTION_NAME)
#define PROFILE_BEGIN_NAMED(name)\
	profile::begin_period(name)
#define PROFILE_END()\
	profile::end_period()

#define PROFILE_SCOPED()\
	profile::ScopedBlock PROFILE_MACRO_PASTE(profile_scoped_, __LINE__)(PROFILE_FUNCTION_NAME)
#define PROFILE_SCOPED_NAMED(name)\
	profile::ScopedBlock PROFILE_MACRO_PASTE(profile_scoped_, __LINE__)(name)

#define PROFILE_THREAD_ENTER()\
	profile::enter_thread(PROFILE_FUNCTION_NAME)
#define PROFILE_THREAD_ENTER_NAMED(name)\
	profile::enter_thread(name)
#define PROFILE_THREAD_EXIT()\
	profile::exit_thread()

#else

#define PROFILE_BEGIN_NAMED(name)
#define PROFILE_BEGIN()
#define PROFILE_END()

#define PROFILE_SCOPED()
#define PROFILE_SCOPED_NAMED(name)

#define PROFILE_THREAD_ENTER()
#define PROFILE_THREAD_ENTER_NAMED(name)
#define PROFILE_THREAD_EXIT()

#endif

namespace profile {

void begin_period(const char* name);
void end_period();
void pause_period();
void unpause_period();
void enter_thread(const char* name);
void exit_thread();
void reset_thread();
void cleanup();

struct ScopedBlock
{
	ScopedBlock(const char* name)
	{
		PROFILE_BEGIN_NAMED(name);
	}
	~ScopedBlock()
	{
		PROFILE_END();
	}
};

} // namespace profile

// §1.8 Immediate Mode Drawing Declarations.....................................

struct Vector3;
struct Vector4;
struct AABB;
struct Triangle;

namespace immediate {

void draw();
void add_line(Vector3 start, Vector3 end, Vector4 colour);
void add_wire_aabb(AABB aabb, Vector4 colour);
void add_triangle(Triangle* triangle, Vector4 colour);

} // namespace immediate

// §1.9 Random Number Generation................................................

// This has been named arandom rather than random because random() is the name
// of a function in stdlib.h under POSIX. Thanks POSIX.
namespace arandom {

/*  Written in 2015 by Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright and
related and neighboring rights to this software to the public domain worldwide.
This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

static uint64_t splitmix64(uint64_t* x)
{
	*x += UINT64_C(0x9E3779B97F4A7C15);
	uint64_t z = *x;
	z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
	z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
	return z ^ (z >> 31);
}

/*  Written in 2016 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright and
related and neighboring rights to this software to the public domain worldwide.
This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

struct Sequence
{
	u64 s[2];
	u64 seed;
};

static inline u64 rotl(const u64 x, int k)
{
	return (x << k) | (x >> (64 - k));
}

// Xoroshiro128+
u64 generate(Sequence* sequence)
{
	const u64 s0 = sequence->s[0];
	u64 s1 = sequence->s[1];
	const u64 result = s0 + s1;

	s1 ^= s0;
	sequence->s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
	sequence->s[1] = rotl(s1, 36); // c

	return result;
}

// End of Blackman & Vigna's code

u64 seed(Sequence* sequence, u64 value)
{
	u64 old_seed = sequence->seed;
	sequence->seed = value;
	sequence->s[0] = splitmix64(&sequence->seed);
	sequence->s[1] = splitmix64(&sequence->seed);
	return old_seed;
}

u64 seed_by_time(Sequence* sequence)
{
	return seed(sequence, get_timestamp());
}

int int_range(Sequence* sequence, int min, int max)
{
	int x = generate(sequence) % static_cast<u64>(max - min + 1);
	return min + x;
}

static inline float to_float(u64 x)
{
	union
	{
		u32 i;
		float f;
	} u;
	u.i = UINT32_C(0x7f) << 23 | x >> 41;
	return u.f - 1.0f;
}

float float_range(Sequence* sequence, float min, float max)
{
	float f = to_float(generate(sequence));
	return min + f * (max - min);
}

} // namespace arandom

// §1.10 Vectors................................................................

#include <cfloat>
#include <climits>

const float infinity = FLT_MAX;

struct Vector2
{
	float x, y;
};

static const Vector2 vector2_zero = {0.0f, 0.0f};

Vector2 operator + (Vector2 v0, Vector2 v1)
{
	Vector2 result;
	result.x = v0.x + v1.x;
	result.y = v0.y + v1.y;
	return result;
}

Vector2& operator += (Vector2& v0, Vector2 v1)
{
	v0.x += v1.x;
	v0.y += v1.y;
	return v0;
}

Vector2 operator - (Vector2 v0, Vector2 v1)
{
	Vector2 result;
	result.x = v0.x - v1.x;
	result.y = v0.y - v1.y;
	return result;
}

Vector2& operator -= (Vector2& v0, Vector2 v1)
{
	v0.x -= v1.x;
	v0.y -= v1.y;
	return v0;
}

Vector2 operator * (Vector2 v, float s)
{
	Vector2 result;
	result.x = v.x * s;
	result.y = v.y * s;
	return result;
}

Vector2 operator * (float s, Vector2 v)
{
	Vector2 result;
	result.x = s * v.x;
	result.y = s * v.y;
	return result;
}

Vector2& operator *= (Vector2& v, float s)
{
	v.x *= s;
	v.y *= s;
	return v;
}

Vector2 operator / (Vector2 v, float s)
{
	Vector2 result;
	result.x = v.x / s;
	result.y = v.y / s;
	return result;
}

Vector2& operator /= (Vector2& v, float s)
{
	v.x /= s;
	v.y /= s;
	return v;
}

Vector2 operator - (Vector2 v)
{
	return {-v.x, -v.y};
}

Vector2 perp(Vector2 v)
{
	return {v.y, -v.x};
}

float squared_length(Vector2 v)
{
	return (v.x * v.x) + (v.y * v.y);
}

float length(Vector2 v)
{
	return sqrt(squared_length(v));
}

Vector2 normalise(Vector2 v)
{
	float l = length(v);
	ASSERT(l != 0.0f && isfinite(l));
	return v / l;
}

float dot(Vector2 v0, Vector2 v1)
{
	return (v0.x * v1.x) + (v0.y * v1.y);
}

Vector2 pointwise_multiply(Vector2 v0, Vector2 v1)
{
	Vector2 result;
	result.x = v0.x * v1.x;
	result.y = v0.y * v1.y;
	return result;
}

Vector2 lerp(Vector2 v0, Vector2 v1, float t)
{
	Vector2 result;
	result.x = lerp(v0.x, v1.x, t);
	result.y = lerp(v0.y, v1.y, t);
	return result;
}

static Vector2 min2(Vector2 a, Vector2 b)
{
	Vector2 result;
	result.x = fmin(a.x, b.x);
	result.y = fmin(a.y, b.y);
	return result;
}

static Vector2 max2(Vector2 a, Vector2 b)
{
	Vector2 result;
	result.x = fmax(a.x, b.x);
	result.y = fmax(a.y, b.y);
	return result;
}

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
const Vector3 vector3_min    = {-infinity, -infinity, -infinity};
const Vector3 vector3_max    = {+infinity, +infinity, +infinity};

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

Vector3 operator + (Vector3 v)
{
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

float distance(Vector3 v0, Vector3 v1)
{
	return length(v1 - v0);
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

Vector3 pointwise_multiply(Vector3 v0, Vector3 v1)
{
	Vector3 result;
	result.x = v0.x * v1.x;
	result.y = v0.y * v1.y;
	result.z = v0.z * v1.z;
	return result;
}

Vector3 pointwise_divide(Vector3 v0, Vector3 v1)
{
	Vector3 result;
	result.x = v0.x / v1.x;
	result.y = v0.y / v1.y;
	result.z = v0.z / v1.z;
	return result;
}

Vector3 lerp(Vector3 v0, Vector3 v1, float t)
{
	Vector3 result;
	result.x = lerp(v0.x, v1.x, t);
	result.y = lerp(v0.y, v1.y, t);
	result.z = lerp(v0.z, v1.z, t);
	return result;
}

Vector3 reciprocal(Vector3 v)
{
	ASSERT(v.x != 0.0f && v.y != 0.0f && v.z != 0.0f);
	Vector3 result;
	result.x = 1.0f / v.x;
	result.y = 1.0f / v.y;
	result.z = 1.0f / v.z;
	return result;
}

static Vector3 max3(Vector3 v0, Vector3 v1)
{
	Vector3 result;
	result.x = fmax(v0.x, v1.x);
	result.y = fmax(v0.y, v1.y);
	result.z = fmax(v0.z, v1.z);
	return result;
}

static Vector3 min3(Vector3 v0, Vector3 v1)
{
	Vector3 result;
	result.x = fmin(v0.x, v1.x);
	result.y = fmin(v0.y, v1.y);
	result.z = fmin(v0.z, v1.z);
	return result;
}

static Vector3 make_vector3(Vector2 v)
{
	return {v.x, v.y, 0.0f};
}

struct Vector4
{
	float x;
	float y;
	float z;
	float w;
};

static Vector4 make_vector4(Vector3 v)
{
	return {v.x, v.y, v.z, 0.0f};
}

// §1.11 Quaternions............................................................

struct Quaternion
{
	float w, x, y, z;
};

const Quaternion quaternion_identity = {1.0f, 0.0f, 0.0f, 0.0f};

float norm(Quaternion q)
{
	float result = sqrt((q.w * q.w) + (q.x * q.x) + (q.y * q.y) + (q.z * q.z));
	return result;
}

Quaternion conjugate(Quaternion q)
{
	Quaternion result;
	result.w = q.w;
	result.x = -q.x;
	result.y = -q.y;
	result.z = -q.z;
	return result;
}

Vector3 operator * (Quaternion q, Vector3 v)
{
    Vector3 vector_part = {q.x, q.y, q.z};
    Vector3 t = 2.0f * cross(vector_part, v);
    return v + (q.w * t) + cross(vector_part, t);
}

Quaternion& operator /= (Quaternion& q, float s)
{
	q.w /= s;
	q.x /= s;
	q.y /= s;
	q.z /= s;
	return q;
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
	ASSERT(almost_one(norm(result)));
	return result;
}

// §1.12 Matrices...............................................................

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

// This transforms from a right-handed coordinate system to OpenGL's default
// clip space. A position will be viewable in this clip space if its x, y,
// and z components are in the range [-w,w] of its w component.
Matrix4 orthographic_projection_matrix(float width, float height, float near_plane, float far_plane)
{
	float neg_depth = near_plane - far_plane;

	Matrix4 result;

	result[0] = 2.0f / width;
	result[1] = 0.0f;
	result[2] = 0.0f;
	result[3] = 0.0f;

	result[4] = 0.0f;
	result[5] = 2.0f / height;
	result[6] = 0.0f;
	result[7] = 0.0f;

	result[8] = 0.0f;
	result[9] = 0.0f;
	result[10] = 2.0f / neg_depth;
	result[11] = (far_plane + near_plane) / neg_depth;

	result[12] = 0.0f;
	result[13] = 0.0f;
	result[14] = 0.0f;
	result[15] = 1.0f;

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
	result[1]  = (       2.0f * (xy + zw)) * scale.y;
	result[2]  = (       2.0f * (xz - yw)) * scale.z;
	result[3]  = position.x;

	result[4]  = (       2.0f * (xy - zw)) * scale.x;
	result[5]  = (1.0f - 2.0f * (xx + zz)) * scale.y;
	result[6]  = (       2.0f * (yz + xw)) * scale.z;
	result[7]  = position.y;

	result[8]  = (       2.0f * (xz + yw)) * scale.x;
	result[9]  = (       2.0f * (yz - xw)) * scale.y;
	result[10] = (1.0f - 2.0f * (xx + yy)) * scale.z;
	result[11] = position.z;

	result[12] = 0.0f;
	result[13] = 0.0f;
	result[14] = 0.0f;
	result[15] = 1.0f;

	return result;
}

Matrix4 inverse_view_matrix(const Matrix4& m)
{
	float a = -((m[0] * m[3]) + (m[4] * m[7]) + (m[8]  * m[11]));
	float b = -((m[1] * m[3]) + (m[5] * m[7]) + (m[9]  * m[11]));
	float c = -((m[2] * m[3]) + (m[6] * m[7]) + (m[10] * m[11]));
	return
	{{
		m[0], m[4], m[8],  a,
		m[1], m[5], m[9],  b,
		m[2], m[6], m[10], c,
		0.0f, 0.0f, 0.0f,  1.0f
	}};
}

Matrix4 inverse_perspective_matrix(const Matrix4& m)
{
	float m0  = 1.0f / m[0];
	float m5  = 1.0f / m[5];
	float m14 = 1.0f / m[11];
	float m15 = m[10] / m[11];
	return
	{{
		m0,   0.0f, 0.0f,  0.0f,
		0.0f, m5,   0.0f,  0.0f,
		0.0f, 0.0f, 0.0f, -1.0f,
		0.0f, 0.0f, m14,   m15
	}};
}

Matrix4 inverse_transform(const Matrix4& m)
{
	// The scale can be extracted from the rotation data by just taking the
	// length of the first three row vectors.

	float dx = sqrt(m[0] * m[0] + m[4] * m[4] + m[8]  * m[8]);
	float dy = sqrt(m[1] * m[1] + m[5] * m[5] + m[9]  * m[9]);
	float dz = sqrt(m[2] * m[2] + m[6] * m[6] + m[10] * m[10]);

	// The extracted scale can then be divided out to isolate the rotation rows.

	float m00 = m[0] / dx;
	float m10 = m[4] / dx;
	float m20 = m[8] / dx;

	float m01 = m[1] / dy;
	float m11 = m[5] / dy;
	float m21 = m[9] / dy;

	float m02 = m[2] / dz;
	float m12 = m[6] / dz;
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
	m10 /= dx;
	m20 /= dx;

	m01 /= dy;
	m11 /= dy;
	m21 /= dy;

	m02 /= dz;
	m12 /= dz;
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

// §2.1 Geometry Functions......................................................

struct LineSegment
{
	Vector2 vertices[2];
};

struct Triangle
{
	// assumes clockwise winding for the front face
	Vector3 vertices[3];
};

struct Quad
{
	// assumes clockwise winding for the front face
	// also, there's nothing guaranteeing these are coplanar
	Vector3 vertices[4];
};

struct Ray
{
	Vector3 origin;
	Vector3 direction;
	Vector3 inverse_direction;
};

Ray make_ray(Vector3 origin, Vector3 direction)
{
	Vector3 normal = normalise(direction);
	Ray result;
	result.origin = origin;
	result.direction = normal;
	result.inverse_direction = reciprocal(normal);
	return result;
}

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
	return
		abs(t.x) <= e0.x + e1.x &&
		abs(t.y) <= e0.y + e1.y &&
		abs(t.z) <= e0.z + e1.z;
}

static bool aabb_validate(AABB b)
{
	// Check that the box actually has positive volume.
	return
		b.max.x > b.min.x &&
		b.max.y > b.min.y &&
		b.max.z > b.min.z;
}

static AABB aabb_from_triangle(Triangle* triangle)
{
	AABB result;
	Vector3 v0 = triangle->vertices[0];
	Vector3 v1 = triangle->vertices[1];
	Vector3 v2 = triangle->vertices[2];
	result.min = min3(min3(v0, v1), v2);
	result.max = max3(max3(v0, v1), v2);
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
	result.min = min3(o0.min, o1.min);
	result.max = max3(o0.max, o1.max);
	return result;
}

static AABB aabb_clip(AABB to, AABB from)
{
	AABB result;
	result.min = max3(to.min, from.min);
	result.max = min3(to.max, from.max);
	return result;
}

static AABB compute_bounds(Triangle* triangles, int lower, int upper)
{
	AABB result = {vector3_max, vector3_min};
	for(int i = lower; i <= upper; ++i)
	{
		AABB aabb = aabb_from_triangle(&triangles[i]);
		result = aabb_merge(result, aabb);
	}
	return result;
}

static AABB compute_bounds(Vector3* positions, int count)
{
	AABB result = {vector3_max, vector3_min};
	for(int i = 0; i < count; ++i)
	{
		result.min = min3(result.min, positions[i]);
		result.max = max3(result.max, positions[i]);
	}
	return result;
}

static float distance_point_to_aabb(AABB a, Vector3 p)
{
	Vector3 v = max3(max3(a.min - p, vector3_zero), p - a.max);
	return length(v);
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

	return *t0 >= 0.0f && *t1 <= 1.0f && *t0 <= *t1;
}

static bool intersect_aabb_ray(AABB aabb, Ray ray)
{
	float t1 = (aabb.min.x - ray.origin.x) * ray.inverse_direction.x;
	float t2 = (aabb.max.x - ray.origin.x) * ray.inverse_direction.x;

	float tl = fmin(t1, t2);
	float th = fmax(t1, t2);

	for(int i = 1; i < 3; ++i)
	{
		t1 = (aabb.min[i] - ray.origin[i]) * ray.inverse_direction[i];
		t2 = (aabb.max[i] - ray.origin[i]) * ray.inverse_direction[i];

		tl = fmax(tl, fmin(fmin(t1, t2), th));
		th = fmin(th, fmax(fmax(t1, t2), tl));
	}

	return th > fmax(tl, 0.0f);
}

// §2.2 Bounding Volume Hierarchy...............................................
namespace bvh {

struct Node
{
	AABB aabb;
	union
	{
		struct
		{
			s32 left;
			s32 right;
		};
		void* user_data;
	};
	union
	{
		s32 parent;
		s32 next;
	};
	s32 height;
};

const s32 null_index = -1;

static bool is_leaf(Node* node)
{
	return node->height == 0;
}

struct Tree
{
	Node* nodes;
	int nodes_capacity;
	int nodes_count;
	s32 free_list;
	s32 root_index;
};

static void initialise_free_list(Tree* tree, s32 index)
{
	for(s32 i = index; i < tree->nodes_capacity - 1; ++i)
	{
		tree->nodes[i].next = i + 1;
		tree->nodes[i].height = null_index;
	}
	Node* last = &tree->nodes[tree->nodes_capacity - 1];
	last->next = null_index;
	last->height = null_index;
	tree->free_list = index;
}

void tree_create(Tree* tree)
{
	tree->nodes_capacity = 128;
	tree->nodes = ALLOCATE(Node, tree->nodes_capacity);
	initialise_free_list(tree, 0);
}

void tree_destroy(Tree* tree)
{
	SAFE_DEALLOCATE(tree->nodes);
}

static s32 allocate_node(Tree* tree)
{
	int prior_capacity = tree->nodes_capacity;
	bool ensured = ENSURE_ARRAY_SIZE(tree->nodes, 1);
	if(!ensured)
	{
		return null_index;
	}
	if(prior_capacity != tree->nodes_capacity)
	{
		initialise_free_list(tree, tree->nodes_count);
	}

	s32 index = tree->free_list;
	tree->free_list = tree->nodes[index].next;
	tree->nodes_count += 1;
	return index;
}

static void deallocate_node(Tree* tree, s32 index)
{
	tree->nodes[index].height = null_index;
	tree->nodes[index].next = tree->free_list;
	tree->free_list = index;
	tree->nodes_count -= 1;
}

static s32 balance_node(Tree* tree, s32 node_index)
{
	// Relevant nodes are lettered as follows:
	//      a
	//     ╱ ╲
	//    ╱   ╲
	//   b     c
	//  ╱ ╲   ╱ ╲
	// d   e f   g

	// Only an inner node with grandchildren can be rotated.
	s32 ai = node_index;
	Node* a = &tree->nodes[ai];
	if(is_leaf(a) || a->height < 2)
	{
		return ai;
	}

	s32 bi = a->left;
	s32 ci = a->right;
	Node* b = &tree->nodes[bi];
	Node* c = &tree->nodes[ci];

	s32 balance = c->height - b->height;
	if(balance > 1)
	{
		// Rotate left to create this layout:
		//     c
		//    ╱ ╲
		//   a   g
		//  ╱ ╲
		// b   f
		// d and e aren't relevant to this move.

		s32 fi = c->left;
		s32 gi = c->right;
		Node* f = &tree->nodes[fi];
		Node* g = &tree->nodes[gi];

		c->left = ai;
		c->parent = a->parent;
		a->parent = ci;

		if(c->parent != null_index)
		{
			if(tree->nodes[c->parent].left == ai)
			{
				tree->nodes[c->parent].left = ci;
			}
			else
			{
				tree->nodes[c->parent].right = ci;
			}
		}
		else
		{
			tree->root_index = ci;
		}

		if(f->height > g->height)
		{
			c->right = fi;
			a->right = gi;
			g->parent = ai;
			a->aabb = aabb_merge(b->aabb, g->aabb);
			c->aabb = aabb_merge(a->aabb, f->aabb);

			a->height = 1 + MAX(b->height, g->height);
			c->height = 1 + MAX(a->height, f->height);
		}
		else
		{
			c->right = gi;
			a->right = fi;
			f->parent = ai;
			a->aabb = aabb_merge(b->aabb, f->aabb);
			c->aabb = aabb_merge(a->aabb, g->aabb);

			a->height = 1 + MAX(b->height, f->height);
			c->height = 1 + MAX(a->height, g->height);
		}

		return ci;
	}
	else if(balance < -1)
	{
		// Rotate right to create this layout:
		//   b
		//  ╱ ╲
		// d   a
		//    ╱ ╲
		//   e   c
		// f and g aren't relevant to this move.

		s32 di = b->left;
		s32 ei = b->right;
		Node* d = &tree->nodes[di];
		Node* e = &tree->nodes[ei];

		b->left = ai;
		b->parent = a->parent;
		a->parent = bi;

		if(b->parent != null_index)
		{
			if(tree->nodes[b->parent].left == ai)
			{
				tree->nodes[b->parent].left = bi;
			}
			else
			{
				tree->nodes[b->parent].right = bi;
			}
		}
		else
		{
			tree->root_index = bi;
		}

		if(d->height > e->height)
		{
			b->right = di;
			a->left = ei;
			e->parent = ai;
			a->aabb = aabb_merge(c->aabb, e->aabb);
			b->aabb = aabb_merge(a->aabb, d->aabb);

			a->height = 1 + MAX(c->height, e->height);
			b->height = 1 + MAX(a->height, d->height);
		}
		else
		{
			b->right = ei;
			a->left = di;
			d->parent = ai;
			a->aabb = aabb_merge(c->aabb, d->aabb);
			b->aabb = aabb_merge(a->aabb, e->aabb);

			a->height = 1 + MAX(c->height, d->height);
			b->height = 1 + MAX(a->height, e->height);
		}

		return bi;
	}
	else
	{
		// The node is already balanced.
		return node_index;
	}
}

static void refresh_parent_chain(Tree* tree, s32 index)
{
	// Walk up from parent to parent, re-balancing each according to its height
	// in the hierarchy, then updating its bounding box and height.
	while(index != null_index)
	{
		index = balance_node(tree, index);

		Node* node = &tree->nodes[index];
		Node* left = &tree->nodes[node->left];
		Node* right = &tree->nodes[node->right];

		node->height = 1 + MAX(left->height, right->height);
		node->aabb = aabb_merge(left->aabb, right->aabb);

		index = node->parent;
	}
}

static float aabb_proximity(AABB a, AABB b)
{
	const Vector3 d = (a.min + a.max) - (b.min + b.max);
	return abs(d.x) + abs(d.y) + abs(d.z);
}

static bool aabb_choose(AABB aabb, AABB c0, AABB c1)
{
	return aabb_proximity(aabb, c0) < aabb_proximity(aabb, c1);
}

s32 insert_node(Tree* tree, AABB aabb, void* user_data)
{
	s32 node_index = allocate_node(tree);
	Node* node = &tree->nodes[node_index];
	node->aabb = aabb;
	node->user_data = user_data;

	// If the tree is empty.
	if(tree->root_index == null_index)
	{
		tree->root_index = node_index;
		node->parent = null_index;
		node->height = 0;
		return node_index;
	}

	// If the root is the only node in the tree, make the root a sibling to the
	// inserted node and create a new root above them.
	Node* root = &tree->nodes[tree->root_index];
	if(is_leaf(root))
	{
		Node* sibling = root;
		s32 sibling_index = tree->root_index;

		s32 root_index = allocate_node(tree);
		Node* root = &tree->nodes[root_index];
		root->aabb = aabb_merge(sibling->aabb, node->aabb);
		root->left = sibling_index;
		root->right = node_index;
		root->parent = null_index;
		root->height = 0;

		sibling->parent = root_index;
		node->parent = root_index;
		tree->root_index = root_index;

		return node_index;
	}

	// Choose the leaf to insert the new node at.
	Node* branch = root;
	s32 branch_index = tree->root_index;
	do
	{
		AABB left_bounds = tree->nodes[branch->left].aabb;
		AABB right_bounds = tree->nodes[branch->right].aabb;
		if(aabb_choose(node->aabb, left_bounds, right_bounds))
		{
			branch_index = branch->left;
		}
		else
		{
			branch_index = branch->right;
		}
		branch = &tree->nodes[branch_index];
	} while(!is_leaf(branch));

	// Replace the branch with a new parent, whose children are the branch and
	// the inserted node.
	s32 parent_index = allocate_node(tree);
	Node* parent = &tree->nodes[parent_index];
	parent->left = branch_index;
	parent->right = node_index;
	root->parent = parent_index;
	node->parent = parent_index;
	refresh_parent_chain(tree, parent_index);

	return node_index;
}

void remove_node(Tree* tree, s32 index)
{
	Node* node = &tree->nodes[index];

	// If this is the only node in the tree.
	if(node->parent == null_index)
	{
		deallocate_node(tree, index);
		tree->root_index = null_index;
		return;
	}

	// All leaves must have a sibling.
	Node* parent = &tree->nodes[node->parent];
	s32 sibling_index;
	if(parent->left == index)
	{
		sibling_index = parent->right;
	}
	else
	{
		sibling_index = parent->left;
	}

	// If this node is a child of the root, make its sibling the new root, and
	// deallocate the node and the old root.
	if(parent->parent == null_index)
	{
		tree->root_index = sibling_index;
		tree->nodes[sibling_index].parent = null_index;
		deallocate_node(tree, index);
		deallocate_node(tree, node->parent);
		return;
	}

	// Replace the parent with the node's sibling and get rid of the replaced
	// parent and the original node.
	Node* grandparent = &tree->nodes[parent->parent];
	if(grandparent->left == node->parent)
	{
		grandparent->left = sibling_index;
	}
	else
	{
		grandparent->right = sibling_index;
	}
	tree->nodes[sibling_index].parent = parent->parent;
	deallocate_node(tree, index);
	deallocate_node(tree, node->parent);
	refresh_parent_chain(tree, sibling_index);
}

s32 update_node(Tree* tree, s32 index, AABB aabb, void* user_data)
{
	remove_node(tree, index);
	return insert_node(tree, aabb, user_data);
}

} // namespace bvh

// §2.3 Bounding Interval Hierarchy.............................................
namespace bih {

enum Flag
{
	FLAG_X,
	FLAG_Y,
	FLAG_Z,
	FLAG_LEAF,
};

struct Node
{
	union
	{
		float clip[2];
		u32 items; // used only by leaf nodes
	};
	u32 index : 30;
	u32 flag : 2;
};

struct Tree
{
	AABB bounds;
	Node* nodes;
	int nodes_count;
	int nodes_capacity;
};

static void tree_destroy(Tree* tree)
{
	SAFE_DEALLOCATE(tree->nodes);
}

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
	float result = +infinity;
	for(int i = lower; i <= upper; ++i)
	{
		Triangle* triangle = &triangles[i];
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
	float result = -infinity;
	for(int i = lower; i <= upper; ++i)
	{
		Triangle* triangle = &triangles[i];
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
		Triangle* triangle = &triangles[pivot];
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

static bool build_node(Tree* tree, Node* node, AABB bounds, Triangle* triangles, int lower, int upper)
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
		Node* children = &tree->nodes[index];

		Node* child_left = &children[0];
		int left_upper = MAX(lower, pivot - 1);
		AABB left_bounds = aabb;
		left_bounds.max[axis] = split;
		node->clip[0] = compute_just_max(triangles, lower, left_upper, axis);
		bool built0 = build_node(tree, child_left, left_bounds, triangles, lower, left_upper);

		Node* child_right = &children[1];
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
	Node* root = &tree->nodes[index];
	tree->bounds = compute_bounds(triangles, 0, triangles_count - 1);

	return build_node(tree, root, tree->bounds, triangles, 0, triangles_count - 1);
}

struct IntersectionResult
{
	int* indices;
	int indices_count;
	int indices_capacity;
};

static bool intersect_node(Node* nodes, Node* node, AABB node_bounds, AABB aabb, Vector3 velocity, IntersectionResult* result)
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
#if 0
		immediate::add_wire_aabb(node_bounds, {1.0f, 1.0f, 1.9f});
#endif
		return true;
	}

	Flag axis = static_cast<Flag>(node->flag);
	Node* children = &nodes[node->index];

	Node* left = &children[0];
	AABB left_bounds = node_bounds;
	left_bounds.max[axis] = node->clip[0];
	bool intersects0 = intersect_node(nodes, left, left_bounds, aabb, velocity, result);

	Node* right = &children[1];
	AABB right_bounds = node_bounds;
	right_bounds.min[axis] = node->clip[1];
	bool intersects1 = intersect_node(nodes, right, right_bounds, aabb, velocity, result);

	return intersects0 || intersects1;
}

bool intersect_tree(Tree* tree, AABB aabb, Vector3 velocity, IntersectionResult* result)
{
	return intersect_node(tree->nodes, tree->nodes, tree->bounds, aabb, velocity, result);
}

static bool intersect_node(Node* nodes, Node* node, AABB bounds, Ray ray, IntersectionResult* result)
{
	bool intersects = intersect_aabb_ray(bounds, ray);
	if(!intersects)
	{
		return false;
	}

	if(node->flag == FLAG_LEAF)
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

	Flag axis = static_cast<Flag>(node->flag);
	Node* children = &nodes[node->index];

	Node* left = &children[0];
	AABB left_bounds = bounds;
	left_bounds.max[axis] = node->clip[0];
	bool intersects0 = intersect_node(nodes, left, left_bounds, ray, result);

	Node* right = &children[1];
	AABB right_bounds = bounds;
	right_bounds.min[axis] = node->clip[1];
	bool intersects1 = intersect_node(nodes, right, right_bounds, ray, result);

	return intersects0 || intersects1;
}

bool intersect_tree(Tree* tree, Ray ray, IntersectionResult* result)
{
	return intersect_node(tree->nodes, tree->nodes, tree->bounds, ray, result);
}

} // namespace bih

// §2.4 Collision Functions.....................................................

struct Collider
{
	bih::Tree tree;
	Triangle* triangles;
	int triangles_count;
};

static void collider_destroy(Collider* collider)
{
	bih::tree_destroy(&collider->tree);
	SAFE_DEALLOCATE(collider->triangles);
}

struct CollisionPacket
{
	Vector3 intersection_point;
	float nearest_distance;
	bool found_collision;
};

static bool point_in_triangle(Vector3 point, Vector3 pa, Vector3 pb, Vector3 pc)
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

static bool get_lowest_root(float a, float b, float c, float max_r, float* root)
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
static float signed_distance(Vector3 point, Vector3 origin, Vector3 normal)
{
	return dot(point, normal) - dot(origin, normal);
}

static void set_packet(CollisionPacket* packet, Vector3 collision_point, float t, Vector3 velocity)
{
	float distance = t * length(velocity);
	if(!packet->found_collision || distance < packet->nearest_distance)
	{
		packet->nearest_distance = distance;
		packet->intersection_point = collision_point;
		packet->found_collision = true;
	}
}

static void collide_unit_sphere_with_triangle(Vector3 center, Vector3 velocity, Triangle triangle, CollisionPacket* packet)
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
	Collider colliders[1];
	int colliders_count;
};

static void check_collision(Vector3 position, Vector3 radius, Vector3 velocity, World* world, CollisionPacket* packet)
{
	Vector3 world_position = pointwise_multiply(position, radius);
	Vector3 world_velocity = pointwise_multiply(velocity, radius);
	AABB aabb = aabb_from_ellipsoid(world_position, radius);
	for(int i = 0; i < world->colliders_count; ++i)
	{
		Collider* collider = &world->colliders[i];
		bih::IntersectionResult intersection = {};
		bih::intersect_tree(&collider->tree, aabb, world_velocity, &intersection);
		for(int j = 0; j < intersection.indices_count; ++j)
		{
			int index = intersection.indices[j];
			Triangle triangle = collider->triangles[index];
			// Transform the triangle's vertex positions to ellipsoid space.
			for(int k = 0; k < 3; ++k)
			{
				Vector3 v = triangle.vertices[k];
				v = pointwise_divide(v, radius);
				triangle.vertices[k] = v;
			}
			collide_unit_sphere_with_triangle(position, velocity, triangle, packet);
#if 0
			// Debug draw each triangle that is checked against.
			// immediate::draw() cannot be called here, so just buffer the
			// primitives and hope they're drawn during the render cycle.
			{
				triangle = collider->triangles[index];
				Vector3 offset = {0.0f, 0.0f, 0.03f};
				for(int l = 0; l < 3; ++l)
				{
					triangle.vertices[l] += offset;
				}
				Vector3 colour = {1.0f, 1.0f, 1.0f};
				immediate::add_triangle(&triangle, colour);
			}
#endif
		}
		SAFE_DEALLOCATE(intersection.indices);
	}
}

static Vector3 collide_with_world(Vector3 position, Vector3 radius, Vector3 velocity, World* world)
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
	Vector3 e_position = pointwise_divide(position, radius);
	Vector3 e_velocity = pointwise_divide(velocity, radius);
	Vector3 collide_position;
	if(squared_length(e_velocity) != 0.0f)
	{
		collide_position = collide_with_world(e_position, radius, e_velocity, world);
		position = pointwise_multiply(collide_position, radius);
		e_position = collide_position;
	}
	velocity = gravity;

	e_velocity = pointwise_divide(gravity, radius);
	collide_position = collide_with_world(e_position, radius, e_velocity, world);
	Vector3 final_position = pointwise_multiply(collide_position, radius);

	return final_position;
}

static void collide_ray_triangle(Triangle* triangle, Ray* ray, CollisionPacket* packet)
{
	Vector3 v0 = triangle->vertices[0];
	Vector3 v1 = triangle->vertices[1];
	Vector3 v2 = triangle->vertices[2];
	Vector3 e1 = v1 - v0;
	Vector3 e2 = v2 - v0;
	Vector3 normal = cross(ray->direction, e2);

	float d = dot(e1, normal);
	if(d < 1e-8f && d > -1e-8f)
	{
		// The ray is parallel to the plane.
		packet->found_collision = false;
		return;
	}

	float id = 1.0f / d;
	Vector3 s = ray->origin - v0;
	float u = dot(s, normal) * id;
	if(u < 0.0f || u > 1.0f)
	{
		packet->found_collision = false;
		return;
	}

	Vector3 q = cross(normal, e1);
	float v = dot(ray->direction, q) * id;
	if(v < 0.0f || u + v > 1.0f)
	{
		packet->found_collision = false;
		return;
	}

	float t = dot(e2, q) * id;
	if(t < 1e-6f)
	{
		packet->found_collision = false;
	}
	else
	{
		packet->found_collision = true;
		packet->nearest_distance = t;
		packet->intersection_point = ray->origin + t * ray->direction;
	}
}

// §2.5 Frustum Functions.......................................................

struct Frustum
{
	struct Plane
	{
		union
		{
			struct
			{
				float x, y, z, w;
			};
			Vector3 normal;
		};
		Vector3 sign_flip;
	};
	Plane planes[6];
};

static float signed_one(float x)
{
	if(x > 0.0f)
	{
		return 1.0f;
	}
	else
	{
		return -1.0f;
	}
}

// Matrix m can be a projection or view-projection matrix, and the frustum will
// be in eye space or world space respectively.
Frustum make_frustum(Matrix4 m)
{
	Frustum result;
	for(int i = 0; i < 6; ++i)
	{
		Frustum::Plane* plane = &result.planes[i];
		float sign = 2 * (i & 1) - 1;
		int row = 4 * (i / 2);
		plane->x = m[12] + sign * m[0 + row];
		plane->y = m[13] + sign * m[1 + row];
		plane->z = m[14] + sign * m[2 + row];
		plane->w = m[15] + sign * m[3 + row];
		plane->sign_flip = {signed_one(plane->x), signed_one(plane->y), signed_one(plane->z)};
	}
	return result;
}

bool intersect_aabb_frustum(Frustum* frustum, AABB* aabb)
{
	// Division by two is ignored for the center and extents and is instead
	// moved down to the test as a multiplication by two.
	Vector3 center = aabb->max + aabb->min;
	Vector3 extent = aabb->max - aabb->min;
	for(int i = 0; i < 6; ++i)
	{
		Frustum::Plane plane = frustum->planes[i];
		Vector3 sf = pointwise_multiply(extent, plane.sign_flip);
		if(dot(center + sf, plane.normal) <= 2.0f * -plane.w)
		{
			return false;
		}
	}
	return true;
}

// §3.1 OpenGL Function and Type Declarations...................................

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

#define GL_BLEND 0x0BE2
#define GL_COLOR_BUFFER_BIT 0x00004000
#define GL_CULL_FACE 0x0B44
#define GL_DEPTH_BUFFER_BIT 0x00000100
#define GL_DEPTH_TEST 0x0B71
#define GL_FALSE 0
#define GL_FLOAT 0x1406
#define GL_LINEAR 0x2601
#define GL_LINEAR_MIPMAP_LINEAR 0x2703
#define GL_LINEAR_MIPMAP_NEAREST 0x2701
#define GL_LINES 0x0001
#define GL_NEAREST 0x2600
#define GL_NEAREST_MIPMAP_LINEAR 0x2702
#define GL_NEAREST_MIPMAP_NEAREST 0x2700
#define GL_ONE_MINUS_SRC_ALPHA 0x0303
#define GL_RED 0x1903
#define GL_REPEAT 0x2901
#define GL_SRC_ALPHA 0x0302
#define GL_TEXTURE_2D 0x0DE1
#define GL_TEXTURE_MAG_FILTER 0x2800
#define GL_TEXTURE_MIN_FILTER 0x2801
#define GL_TEXTURE_WRAP_S 0x2802
#define GL_TEXTURE_WRAP_T 0x2803
#define GL_TRIANGLES 0x0004
#define GL_TRUE 1
#define GL_UNSIGNED_BYTE 0x1401
#define GL_UNSIGNED_INT 0x1405
#define GL_UNSIGNED_SHORT 0x1403

#define GL_TEXTURE0 0x84C0

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

#define GL_R8 0x8229

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

void (APIENTRYA *p_glBlendFunc)(GLenum sfactor, GLenum dfactor) = nullptr;
void (APIENTRYA *p_glClear)(GLbitfield mask) = nullptr;
void (APIENTRYA *p_glDepthMask)(GLboolean flag) = nullptr;
void (APIENTRYA *p_glDisable)(GLenum cap) = nullptr;
void (APIENTRYA *p_glEnable)(GLenum cap) = nullptr;
void (APIENTRYA *p_glTexImage2D)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void* pixels) = nullptr;
void (APIENTRYA *p_glViewport)(GLint x, GLint y, GLsizei width, GLsizei height) = nullptr;

void (APIENTRYA *p_glBindTexture)(GLenum target, GLuint texture) = nullptr;
void (APIENTRYA *p_glDeleteTextures)(GLsizei n, const GLuint* textures) = nullptr;
void (APIENTRYA *p_glDrawArrays)(GLenum mode, GLint first, GLsizei count) = nullptr;
void (APIENTRYA *p_glDrawElements)(GLenum mode, GLsizei count, GLenum type, const void* indices) = nullptr;
void (APIENTRYA *p_glGenTextures)(GLsizei n, GLuint* textures) = nullptr;
void (APIENTRYA *p_glTexSubImage2D)(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void * pixels) = nullptr;

void (APIENTRYA *p_glActiveTexture)(GLenum texture) = nullptr;

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
void (APIENTRYA *p_glUniform1f)(GLint location, GLfloat v0) = nullptr;
void (APIENTRYA *p_glUniform1i)(GLint location, GLint v0) = nullptr;
void (APIENTRYA *p_glUniform3fv)(GLint location, GLsizei count, const GLfloat* value) = nullptr;
void (APIENTRYA *p_glUniformMatrix4fv)(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value) = nullptr;
void (APIENTRYA *p_glUseProgram)(GLuint program) = nullptr;
void (APIENTRYA *p_glVertexAttribPointer)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer) = nullptr;

void (APIENTRYA *p_glBindSampler)(GLuint unit, GLuint sampler) = nullptr;
void (APIENTRYA *p_glDeleteSamplers)(GLsizei count, const GLuint * samplers) = nullptr;
void (APIENTRYA *p_glGenSamplers)(GLsizei count, GLuint * samplers) = nullptr;
void (APIENTRYA *p_glSamplerParameteri)(GLuint sampler, GLenum pname, GLint param) = nullptr;

#define glBindVertexArray p_glBindVertexArray
#define glDeleteVertexArrays p_glDeleteVertexArrays
#define glGenVertexArrays p_glGenVertexArrays

#define glBindBuffer p_glBindBuffer
#define glBufferData p_glBufferData
#define glDeleteBuffers p_glDeleteBuffers
#define glGenBuffers p_glGenBuffers
#define glMapBuffer p_glMapBuffer
#define glUnmapBuffer p_glUnmapBuffer

#define glBlendFunc p_glBlendFunc
#define glClear p_glClear
#define glDepthMask p_glDepthMask
#define glDisable p_glDisable
#define glEnable p_glEnable
#define glTexImage2D p_glTexImage2D
#define glViewport p_glViewport

#define glBindTexture p_glBindTexture
#define glDeleteTextures p_glDeleteTextures
#define glDrawArrays p_glDrawArrays
#define glDrawElements p_glDrawElements
#define glGenTextures p_glGenTextures
#define glTexSubImage2D p_glTexSubImage2D

#define glActiveTexture p_glActiveTexture

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
#define glUniform1f p_glUniform1f
#define glUniform1i p_glUniform1i
#define glUniform3fv p_glUniform3fv
#define glUniformMatrix4fv p_glUniformMatrix4fv
#define glUseProgram p_glUseProgram
#define glVertexAttribPointer p_glVertexAttribPointer

#define glBindSampler p_glBindSampler
#define glDeleteSamplers p_glDeleteSamplers
#define glGenSamplers p_glGenSamplers
#define glSamplerParameteri p_glSamplerParameteri

// §3.2 Shader Functions........................................................

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

// §3.3 Vertex Types and Packing................................................

union Pack4x8
{
	struct
	{
		u8 r, g, b, a;
	};
	u32 packed;
};

static u32 pack_unorm1x8(float r)
{
	Pack4x8 u;
	u.r = 0xff * r;
	u.g = 0xff * r;
	u.b = 0xff * r;
	u.a = 0xff;
	return u.packed;
}

static u32 pack_unorm3x8(Vector3 v)
{
	Pack4x8 u;
	u.r = 0xff * v.x;
	u.g = 0xff * v.y;
	u.b = 0xff * v.z;
	u.a = 0xff;
	return u.packed;
}

static u32 pack_unorm4x8(Vector4 v)
{
	Pack4x8 u;
	u.r = 0xff * v.x;
	u.g = 0xff * v.y;
	u.b = 0xff * v.z;
	u.a = 0xff * v.w;
	return u.packed;
}

static Vector4 unpack_unorm4x8(u32 x)
{
	Pack4x8 u;
	u.packed = x;
	Vector4 v;
	v.x = u.r / 255.0f;
	v.y = u.g / 255.0f;
	v.z = u.b / 255.0f;
	v.w = u.a / 255.0f;
	return v;
}

static u32 pack_snorm3x10(Vector3 v)
{
	union
	{
		struct
		{
			int x : 10;
			int y : 10;
			int z : 10;
			int w : 2;
		};
		u32 packed;
	} u;
	u.x = round(v.x * 511.0f);
	u.y = round(v.y * 511.0f);
	u.z = round(v.z * 511.0f);
	u.w = 0;
	return u.packed;
}

static u32 pack_unorm16x2(Vector2 v)
{
	union
	{
		struct
		{
			u16 x;
			u16 y;
		};
		u32 packed;
	} u;
	u.x = 0xffff * v.x;
	u.y = 0xffff * v.y;
	return u.packed;
}

static bool is_snorm(float x)
{
	return x >= -1.0f && x <= 1.0f;
}

static bool is_unorm(float x)
{
	return x >= 0.0f && x <= 1.0f;
}

static u32 r_to_u32(float r)
{
	ASSERT(is_unorm(r));
	return pack_unorm1x8(r);
}

static u32 rgb_to_u32(Vector3 c)
{
	ASSERT(is_unorm(c.x));
	ASSERT(is_unorm(c.y));
	ASSERT(is_unorm(c.z));
	return pack_unorm3x8(c);
}

static u32 rgba_to_u32(Vector4 c)
{
	ASSERT(is_unorm(c.x));
	ASSERT(is_unorm(c.y));
	ASSERT(is_unorm(c.z));
	ASSERT(is_unorm(c.w));
	return pack_unorm4x8(c);
}

static Vector4 u32_to_rgba(u32 u)
{
	return unpack_unorm4x8(u);
}

static u32 normal_to_u32(Vector3 v)
{
	ASSERT(is_snorm(v.x));
	ASSERT(is_snorm(v.y));
	ASSERT(is_snorm(v.z));
	return pack_snorm3x10(v);
}

static u32 texcoord_to_u32(Vector2 v)
{
	ASSERT(is_unorm(v.x));
	ASSERT(is_unorm(v.y));
	return pack_unorm16x2(v);
}

struct VertexPC
{
	Vector3 position;
	u32 colour;
};

struct VertexPNC
{
	Vector3 position;
	Vector3 normal;
	u32 colour;
};

struct VertexPT
{
	Vector3 position;
	u32 texcoord;
};

// §3.4 Floor...................................................................

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
static bool floor_add_box(Floor* floor, Vector3 bottom_left, Vector3 dimensions, arandom::Sequence* randomness)
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
	floor->vertices[o     ].normal   = +vector3_unit_z;
	floor->vertices[o +  1].normal   = +vector3_unit_z;
	floor->vertices[o +  2].normal   = +vector3_unit_z;
	floor->vertices[o +  3].normal   = +vector3_unit_z;
	floor->vertices[o +  4].normal   = +vector3_unit_x;
	floor->vertices[o +  5].normal   = +vector3_unit_x;
	floor->vertices[o +  6].normal   = +vector3_unit_x;
	floor->vertices[o +  7].normal   = +vector3_unit_x;
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
		float rando = arandom::float_range(randomness, 0.0f, 1.0f);
		Vector3 colour = {0.0f, 1.0f, rando};
		floor->vertices[o + i].colour = rgb_to_u32(colour);
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

// §3.5 Immediate Mode Drawing..................................................

namespace immediate {

static const int context_vertices_cap = 8192;
static const int context_vertex_type_count = 2;

enum class DrawMode
{
	None,
	Lines,
	Triangles,
};

enum class BlendMode
{
	None,
	Opaque,
	Transparent,
};

enum class VertexType
{
	None,
	Colour,
	Texture,
};

struct Context
{
	union
	{
		VertexPC vertices[context_vertices_cap];
		VertexPT vertices_textured[context_vertices_cap];
	};
	Matrix4 view_projection;
	GLuint vertex_arrays[context_vertex_type_count];
	GLuint buffers[context_vertex_type_count];
	GLuint shaders[context_vertex_type_count];
	int filled;
	DrawMode draw_mode;
	BlendMode blend_mode;
	VertexType vertex_type;
	bool blend_mode_changed;
};

Context* context;

static void context_create()
{
	context = ALLOCATE(Context, 1);
	Context* c = context;

	glGenVertexArrays(context_vertex_type_count, c->vertex_arrays);
	glGenBuffers(context_vertex_type_count, c->buffers);

	glBindVertexArray(c->vertex_arrays[0]);
	glBindBuffer(GL_ARRAY_BUFFER, c->buffers[0]);

	glBufferData(GL_ARRAY_BUFFER, sizeof(c->vertices), nullptr, GL_DYNAMIC_DRAW);

	GLvoid* offset0 = reinterpret_cast<GLvoid*>(offsetof(VertexPC, position));
	GLvoid* offset1 = reinterpret_cast<GLvoid*>(offsetof(VertexPC, colour));
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPC), offset0);
	glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(VertexPC), offset1);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(2);

	glBindVertexArray(c->vertex_arrays[1]);
	glBindBuffer(GL_ARRAY_BUFFER, c->buffers[1]);

	glBufferData(GL_ARRAY_BUFFER, sizeof(c->vertices_textured), nullptr, GL_DYNAMIC_DRAW);

	offset0 = reinterpret_cast<GLvoid*>(offsetof(VertexPT, position));
	offset1 = reinterpret_cast<GLvoid*>(offsetof(VertexPT, texcoord));
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPT), offset0);
	glVertexAttribPointer(3, 2, GL_UNSIGNED_SHORT, GL_TRUE, sizeof(VertexPT), offset1);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(3);

	glBindVertexArray(0);
}

static void context_destroy()
{
	if(context)
	{
		Context* c = context;

		glDeleteBuffers(context_vertex_type_count, c->buffers);
		glDeleteVertexArrays(context_vertex_type_count, c->vertex_arrays);
		DEALLOCATE(c);
	}
}

static void set_matrices(Matrix4 view, Matrix4 projection)
{
	context->view_projection = projection * view;
}

static void set_blend_mode(BlendMode mode)
{
	if(context->blend_mode != mode)
	{
		context->blend_mode = mode;
		context->blend_mode_changed = true;
	}
}

static GLenum get_mode(DrawMode draw_mode)
{
	switch(draw_mode)
	{
		default:
		case DrawMode::Lines:     return GL_LINES;
		case DrawMode::Triangles: return GL_TRIANGLES;
	}
}

void draw()
{
	Context* c = context;
	if(c->filled == 0 || c->draw_mode == DrawMode::None || c->vertex_type == VertexType::None)
	{
		return;
	}

	if(c->blend_mode_changed)
	{
		switch(c->blend_mode)
		{
			case BlendMode::None:
			case BlendMode::Opaque:
			{
				glDisable(GL_BLEND);
				glDepthMask(GL_TRUE);
				glEnable(GL_CULL_FACE);
				break;
			}
			case BlendMode::Transparent:
			{
				glEnable(GL_BLEND);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
				glDepthMask(GL_FALSE);
				glDisable(GL_CULL_FACE);
				break;
			}
		}
		c->blend_mode_changed = false;
	}

	GLuint shader;
	switch(c->vertex_type)
	{
		case VertexType::None:
		case VertexType::Colour:
		{
			glBindBuffer(GL_ARRAY_BUFFER, c->buffers[0]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(VertexPC) * c->filled, c->vertices, GL_DYNAMIC_DRAW);
			glBindVertexArray(c->vertex_arrays[0]);
			shader = c->shaders[0];
			break;
		}
		case VertexType::Texture:
		{
			glBindBuffer(GL_ARRAY_BUFFER, c->buffers[1]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(VertexPT) * c->filled, c->vertices_textured, GL_DYNAMIC_DRAW);
			glBindVertexArray(c->vertex_arrays[1]);
			shader = c->shaders[1];
			break;
		}
	}

	glUseProgram(shader);
	GLint location = glGetUniformLocation(shader, "model_view_projection");
	glUniformMatrix4fv(location, 1, GL_TRUE, c->view_projection.elements);

	glDrawArrays(get_mode(c->draw_mode), 0, c->filled);

	c->draw_mode = DrawMode::None;
	set_blend_mode(BlendMode::None);
	c->vertex_type = VertexType::None;
	c->filled = 0;
}

void add_line(Vector3 start, Vector3 end, Vector4 colour)
{
	Context* c = context;
	ASSERT(c->draw_mode == DrawMode::Lines || c->draw_mode == DrawMode::None);
	ASSERT(c->vertex_type == VertexType::Colour || c->vertex_type == VertexType::None);
	ASSERT(c->filled + 2 < context_vertices_cap);
	u32 colour_u32 = rgba_to_u32(colour);
	c->vertices[c->filled + 0] = {start, colour_u32};
	c->vertices[c->filled + 1] = {end, colour_u32};
	c->filled += 2;
	c->draw_mode = DrawMode::Lines;
	c->vertex_type = VertexType::Colour;
}

void add_line_gradient(Vector3 start, Vector3 end, Vector4 colour0, Vector4 colour1)
{
	Context* c = context;
	ASSERT(c->draw_mode == DrawMode::Lines || c->draw_mode == DrawMode::None);
	ASSERT(c->vertex_type == VertexType::Colour || c->vertex_type == VertexType::None);
	ASSERT(c->filled + 2 < context_vertices_cap);
	c->vertices[c->filled + 0] = {start, rgba_to_u32(colour0)};
	c->vertices[c->filled + 1] = {end, rgba_to_u32(colour1)};
	c->filled += 2;
	c->draw_mode = DrawMode::Lines;
	c->vertex_type = VertexType::Colour;
}

// This is an approximate formula for an ellipse's perimeter. It has the
// greatest error when the ratio of a to b is largest.
static float ellipse_perimeter(float a, float b)
{
	return tau * sqrt(((a * a) + (b * b)) / 2.0f);
}

static void add_wire_ellipse(Vector3 center, Quaternion orientation, Vector2 radius, Vector4 colour)
{
	Context* c = context;

	const float min_spacing = 0.05f;
	float a = radius.x;
	float b = radius.y;
	int segments = ellipse_perimeter(a, b) / min_spacing;
	ASSERT(c->filled + 2 * segments < context_vertices_cap);
	Vector3 point = {a, 0.0f, 0.0f};
	Vector3 position = center + (orientation * point);
	for(int i = 1; i <= segments; ++i)
	{
		Vector3 prior = position;
		float t = (static_cast<float>(i) / segments) * tau;
		point = {a * cos(t), b * sin(t), 0.0f};
		position = center + (orientation * point);
		add_line(prior, position, colour);
	}
}

static void add_wire_ellipsoid(Vector3 center, Vector3 radius, Vector4 colour)
{
	Quaternion q0 = axis_angle_rotation(vector3_unit_y, +pi_over_2);
	Quaternion q1 = axis_angle_rotation(vector3_unit_x, -pi_over_2);
	Quaternion q2 = quaternion_identity;
	add_wire_ellipse(center, q0, {radius.z, radius.y}, colour);
	add_wire_ellipse(center, q1, {radius.x, radius.z}, colour);
	add_wire_ellipse(center, q2, {radius.x, radius.y}, colour);
}

void add_wire_aabb(AABB aabb, Vector4 colour)
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

	add_line(p[0], p[1], colour);
	add_line(p[1], p[3], colour);
	add_line(p[3], p[2], colour);
	add_line(p[2], p[0], colour);

	add_line(p[0], p[4], colour);
	add_line(p[1], p[5], colour);
	add_line(p[2], p[6], colour);
	add_line(p[3], p[7], colour);

	add_line(p[4], p[5], colour);
	add_line(p[5], p[7], colour);
	add_line(p[7], p[6], colour);
	add_line(p[6], p[4], colour);
}

void add_triangle(Triangle* triangle, Vector4 colour)
{
	Context* c = context;
	ASSERT(c->draw_mode == DrawMode::Triangles || c->draw_mode == DrawMode::None);
	ASSERT(c->vertex_type == VertexType::Colour || c->vertex_type == VertexType::None);
	ASSERT(c->filled + 3 < context_vertices_cap);
	for(int i = 0; i < 3; ++i)
	{
		c->vertices[c->filled + i].position = triangle->vertices[i];
		c->vertices[c->filled + i].colour = rgba_to_u32(colour);
	}
	c->filled += 3;
	c->draw_mode = DrawMode::Triangles;
	c->vertex_type = VertexType::Colour;
}

void add_quad(Quad* quad, Vector4 colour)
{
	Context* c = context;
	ASSERT(c->draw_mode == DrawMode::Triangles || c->draw_mode == DrawMode::None);
	ASSERT(c->vertex_type == VertexType::Colour || c->vertex_type == VertexType::None);
	ASSERT(c->filled + 6 < context_vertices_cap);
	c->vertices[c->filled + 0].position = quad->vertices[0];
	c->vertices[c->filled + 1].position = quad->vertices[1];
	c->vertices[c->filled + 2].position = quad->vertices[2];
	c->vertices[c->filled + 3].position = quad->vertices[0];
	c->vertices[c->filled + 4].position = quad->vertices[2];
	c->vertices[c->filled + 5].position = quad->vertices[3];
	for(int i = 0; i < 6; ++i)
	{
		c->vertices[c->filled + i].colour = rgba_to_u32(colour);
	}
	c->filled += 6;
	c->draw_mode = DrawMode::Triangles;
	c->vertex_type = VertexType::Colour;
}

void add_quad_textured(Quad* quad)
{
	Context* c = context;
	ASSERT(c->draw_mode == DrawMode::Triangles || c->draw_mode == DrawMode::None);
	ASSERT(c->vertex_type == VertexType::Texture || c->vertex_type == VertexType::None);
	ASSERT(c->filled + 6 < context_vertices_cap);
	c->vertices_textured[c->filled + 0].position = quad->vertices[0];
	c->vertices_textured[c->filled + 1].position = quad->vertices[1];
	c->vertices_textured[c->filled + 2].position = quad->vertices[2];
	c->vertices_textured[c->filled + 3].position = quad->vertices[0];
	c->vertices_textured[c->filled + 4].position = quad->vertices[2];
	c->vertices_textured[c->filled + 5].position = quad->vertices[3];
	c->vertices_textured[c->filled + 0].texcoord = texcoord_to_u32({0.0f, 0.0f});
	c->vertices_textured[c->filled + 1].texcoord = texcoord_to_u32({1.0f, 0.0f});
	c->vertices_textured[c->filled + 2].texcoord = texcoord_to_u32({1.0f, 1.0f});
	c->vertices_textured[c->filled + 3].texcoord = texcoord_to_u32({0.0f, 0.0f});
	c->vertices_textured[c->filled + 4].texcoord = texcoord_to_u32({1.0f, 1.0f});
	c->vertices_textured[c->filled + 5].texcoord = texcoord_to_u32({0.0f, 1.0f});
	c->filled += 6;
	c->draw_mode = DrawMode::Triangles;
	c->vertex_type = VertexType::Texture;
}

void add_wire_frustum(Matrix4 view, Matrix4 projection, Vector4 colour)
{
	Matrix4 inverse = inverse_view_matrix(view) * inverse_perspective_matrix(projection);

	Vector3 p[8] =
	{
		{-1.0f, -1.0f, -1.0f},
		{-1.0f, -1.0f, +1.0f},
		{-1.0f, +1.0f, -1.0f},
		{-1.0f, +1.0f, +1.0f},
		{+1.0f, -1.0f, -1.0f},
		{+1.0f, -1.0f, +1.0f},
		{+1.0f, +1.0f, -1.0f},
		{+1.0f, +1.0f, +1.0f},
	};
	for(int i = 0; i < 8; ++i)
	{
		p[i] = inverse * p[i];
	}

	add_line(p[0], p[1], colour);
	add_line(p[1], p[5], colour);
	add_line(p[5], p[4], colour);
	add_line(p[4], p[0], colour);

	add_line(p[0], p[2], colour);
	add_line(p[1], p[3], colour);
	add_line(p[4], p[6], colour);
	add_line(p[5], p[7], colour);

	add_line(p[2], p[3], colour);
	add_line(p[3], p[7], colour);
	add_line(p[7], p[6], colour);
	add_line(p[6], p[2], colour);
}

} // namespace immediate

// §3.6 Debug Visualisers.......................................................

static void add_aabb_plane(AABB aabb, bih::Flag axis, float clip, Vector4 colour)
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
			p[2] += {0.0f, d.y , d.z };
			p[3] += {0.0f, 0.0f, d.z };
			break;
		}
		case bih::FLAG_Y:
		{
			p[1] += {d.x , 0.0f, 0.0f};
			p[2] += {d.x , 0.0f, d.z };
			p[3] += {0.0f, 0.0f, d.z };
			break;
		}
		case bih::FLAG_Z:
		{
			p[1] += {d.x , 0.0f, 0.0f};
			p[2] += {d.x , d.y , 0.0f};
			p[3] += {0.0f, d.y , 0.0f};
			break;
		}
		default:
		{
			ASSERT(false);
			break;
		}
	}
	Quad quad;
	for(int i = 0; i < 4; ++i)
	{
		quad.vertices[i] = p[i];
	}
	immediate::add_quad(&quad, colour);
}

static void add_bih_node(bih::Node* nodes, bih::Node* node, AABB bounds, int depth, int target_depth)
{
	if(node->flag != bih::FLAG_LEAF)
	{
		bih::Flag axis = static_cast<bih::Flag>(node->flag);

		AABB left = bounds;
		left.max[axis] = node->clip[0];
		if(depth == target_depth || target_depth < 0)
		{
			add_aabb_plane(bounds, axis, node->clip[0], {1.0f, 0.0f, 0.0f, 0.3f});
		}
		add_bih_node(nodes, &nodes[node->index], left, depth + 1, target_depth);

		AABB right = bounds;
		right.min[axis] = node->clip[1];
		if(depth == target_depth || target_depth < 0)
		{
			add_aabb_plane(bounds, axis, node->clip[1], {0.0f, 0.0f, 1.0f, 0.3f});
		}
		add_bih_node(nodes, &nodes[node->index + 1], right, depth + 1, target_depth);
	}
}

static void draw_bih_tree(bih::Tree* tree, int target_depth)
{
	bih::Node* root = &tree->nodes[0];
	AABB bounds = tree->bounds;
	add_bih_node(tree->nodes, root, bounds, 0, target_depth);
	immediate::set_blend_mode(immediate::BlendMode::Transparent);
	immediate::draw();
}

// §3.7 Colour..................................................................

static const Vector4 colour_white = {1.0f, 1.0f, 1.0f, 1.0f};
static const Vector4 colour_black = {0.0f, 0.0f, 0.0f, 1.0f};
static const Vector4 colour_red = {1.0f, 0.0f, 0.0f, 1.0f};
static const Vector4 colour_green = {0.0f, 1.0f, 0.0f, 1.0f};
static const Vector4 colour_blue = {0.0f, 0.0f, 1.0f, 1.0f};
static const Vector4 colour_cyan = {0.0f, 1.0f, 1.0f, 1.0f};
static const Vector4 colour_magenta = {1.0f, 0.0f, 1.0f, 1.0f};
static const Vector4 colour_yellow = {1.0f, 1.0f, 0.0f, 1.0f};

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
	if(s == 0.0f)
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

// Based on Dave Green's public domain (Unlicense license) Fortran 77
// implementation for cube helix colour table generation.
static void cube_helix(Vector3* colours, int levels, float start_hue, float rotations, float saturation_min, float saturation_max, float lightness_min, float lightness_max, float gamma)
{
	for(int i = 0; i < levels; ++i)
	{
		float fraction = lerp(lightness_min, lightness_max, i / static_cast<float>(levels));
		float saturation = lerp(saturation_min, saturation_max, fraction);
		float angle = tau * (start_hue / 3.0f + 1.0f + rotations * fraction);
		fraction = pow(fraction, gamma);
		float amplitude = saturation * fraction * (1.0f - fraction) / 2.0f;
		float r = -0.14861f * cos(angle) + 1.78277f * sin(angle);
		float g = -0.29227f * cos(angle) - 0.90649f * sin(angle);
		float b = 1.97294f * cos(angle);
		r = fraction + amplitude * r;
		g = fraction + amplitude * g;
		b = fraction + amplitude * b;
		r = clamp(r, 0.0f, 1.0f);
		g = clamp(g, 0.0f, 1.0f);
		b = clamp(b, 0.0f, 1.0f);
		colours[i].x = r;
		colours[i].y = g;
		colours[i].z = b;
	}
}

// §3.8 Text....................................................................

struct Rect
{
	Vector2 bottom_left;
	Vector2 dimensions;
};

static bool clip_test(float q, float p, float* te, float* tl)
{
	if(p == 0.0f)
	{
		return q < 0.0f;
	}
	float t = q / p;
	if(p > 0.0f)
	{
		if(t > *tl)
		{
			return false;
		}
		if(t > *te)
		{
			*te = t;
		}
	}
	else
	{
		if(t < *te)
		{
			return false;
		}
		if(t < *tl)
		{
			*tl = t;
		}
	}
	return true;
}

static bool clip_line(Rect rect, Vector2* p0, Vector2* p1)
{
	// This uses the Liang–Barsky line clipping algorithm.

	float x0 = p0->x;
	float y0 = p0->y;
	float x1 = p1->x;
	float y1 = p1->y;

	// the rectangle's boundaries
	float x_min = rect.bottom_left.x;
	float x_max = rect.bottom_left.x + rect.dimensions.x;
	float y_min = rect.bottom_left.y;
	float y_max = rect.bottom_left.y + rect.dimensions.y;

	// for the line segment (x0, y0) to (x1, y1), derive the parametric form
	// of its line:
	// x = x0 + t * (x1 - x0)
	// y = y0 + t * (y1 - y0)

	float dx = x1 - x0;
	float dy = y1 - y0;

	if((almost_zero(dx) && almost_zero(dy)) && (x0 < x_min || x0 > x_max || y0 < y_min || y0 > y_max))
	{
		// The line is a point and is outside the rectangle.
		return false;
	}

	float te = 0.0f; // entering
	float tl = 1.0f; // leaving
	if(
		clip_test(x_min - x0, +dx, &te, &tl) &&
		clip_test(x0 - x_max, -dx, &te, &tl) &&
		clip_test(y_min - y0, +dy, &te, &tl) &&
		clip_test(y0 - y_max, -dy, &te, &tl))
	{
		if(tl < 1.0f)
		{
			x1 = x0 + (tl * dx);
			y1 = y0 + (tl * dy);
		}
		if(te > 0.0f)
		{
			x0 += te * dx;
			y0 += te * dy;
		}

		p0->x = x0;
		p0->y = y0;
		p1->x = x1;
		p1->y = y1;
		return true;
	}
	else
	{
		// The line must be entirely outside rectangle.
		return false;
	}
}

// 16-Segment Display Numbering:
//
//  -0--|--1-
// |\   9   /|
// | 8  |  10|
// 7  \ | /  2
// --15- -11--
// 6  / | \  3
// | 14 | 12 |
// |/   13  \|
//  -5--|--4-
//
// The number is also the bit number in the 16-segment code. So segment 0
// being on/off is the least significant bit, and 15, the most significant.

// CHEAT ZONE!!! Hardcoded data ahead
static u16 ascii_to_16_segment_table[94] =
{
	0b0000000000001100, // !
	0b0000001000000100, // "
	0b1010101000111100, // #
	0b1010101010111011, // $
	0b1110111010011001, // %
	0b1001001101110001, // &
	0b0000001000000000, // '
	0b0001010000000000, // (
	0b0100000100000000, // )
	0b1111111100000000, // *
	0b1010101000000000, // +
	0b0100000000000000, // ,
	0b1000100000000000, // -
	0b0001000000000000, // .
	0b0100010000000000, // /
	0b0100010011111111, // 0
	0b0000010000001100, // 1
	0b1000100001110111, // 2
	0b0000100000111111, // 3
	0b1000100010001100, // 4
	0b1001000010110011, // 5
	0b1000100011111011, // 6
	0b0000000000001111, // 7
	0b1000100011111111, // 8
	0b1000100010111111, // 9
	0b0000000000110011, // :
	0b0100000000000011, // ;
	0b1001010000000000, // <
	0b1000100000110000, // =
	0b0100100100000000, // >
	0b0010100000000111, // ?
	0b0000101011110111, // @
	0b1000100011001111, // A
	0b0010101000111111, // B
	0b0000000011110011, // C
	0b0010001000111111, // D
	0b1000000011110011, // E
	0b1000000011000011, // F
	0b0000100011111011, // G
	0b1000100011001100, // H
	0b0010001000110011, // I
	0b0000000001111100, // J
	0b1001010011000000, // K
	0b0000000011110000, // L
	0b0000010111001100, // M
	0b0001000111001100, // N
	0b0000000011111111, // O
	0b1000100011000111, // P
	0b0001000011111111, // Q
	0b1001100011000111, // R
	0b1000100010111011, // S
	0b0010001000000011, // T
	0b0000000011111100, // U
	0b0100010011000000, // V
	0b0101000011001100, // W
	0b0101010100000000, // X
	0b0010010100000000, // Y
	0b0100010000110011, // Z
	0b0010001000010010, // [
	0b0001000100000000, /* \ */
	0b0010001000100001, // ]
	0b0101000000000000, // ^
	0b0000000000110000, // _
	0b0000000100000000, // `
	0b1010000001110000, // a
	0b1000100011111000, // b
	0b1000100001110000, // c
	0b1000100001111100, // d
	0b1100000001110000, // e
	0b1010101000000010, // f
	0b1000100010111111, // g
	0b1000100011001000, // h
	0b0010000000000000, // i
	0b0000000000111000, // j
	0b1001100011000000, // k
	0b0000000001110000, // l
	0b1010100001001000, // m
	0b1000100001001000, // n
	0b1000100001111000, // o
	0b1000100011000111, // p
	0b1000100010001111, // q
	0b1000100001000000, // r
	0b0001100000110000, // s
	0b1010101000010000, // t
	0b0000000001111000, // u
	0b0100000001000000, // v
	0b0101000001001000, // w
	0b1101100000000000, // x
	0b1000100010111100, // y
	0b1100000000110000, // z
	0b1010001000010010, // {
	0b0010001000000000, // |
	0b0010101000100001, // }
	0b0000000000000011, // ~
};

static Vector2 segment_lines[16][2] =
{
	{{0.25f, 1.0f}, {0.75f, 1.0f}},
	{{0.75f, 1.0f}, {1.25f, 1.0f}},
	{{1.25f, 1.0f}, {1.125f, 0.5f}},
	{{1.125f, 0.5f}, {1.0f, 0.0f}},
	{{0.5f, 0.0f}, {1.0f, 0.0f}},
	{{0.0f, 0.0f}, {0.5f, 0.0f}},
	{{0.0f, 0.0f}, {0.125f, 0.5f}},
	{{0.125f, 0.5f}, {0.25f, 1.0f}},
	{{0.25f, 1.0f}, {0.625f, 0.5f}},
	{{0.625f, 0.5f}, {0.75f, 1.0f}},
	{{0.625f, 0.5f}, {1.25f, 1.0f}},
	{{0.625f, 0.5f}, {1.125f, 0.5f}},
	{{0.625f, 0.5f}, {1.0f, 0.0f}},
	{{0.5f, 0.0f}, {0.625f, 0.5f}},
	{{0.0f, 0.0f}, {0.625f, 0.5f}},
	{{0.125f, 0.5f}, {0.625f, 0.5f}},
};
// CHEAT ZONE exited, thank god

static u16 ascii_to_16_segment_code(char c)
{
	ASSERT(c >= '!' && c <= '~');
	return ascii_to_16_segment_table[c - '!'];
}

static void draw_16_segment_glyph(u16 code, Rect glyph, Rect clip, Vector4 colour)
{
	for(int i = 0; i < 16; ++i)
	{
		if(code & (1 << i))
		{
			Vector2 s0 = segment_lines[i][0];
			Vector2 s1 = segment_lines[i][1];
			s0 = pointwise_multiply(glyph.dimensions, s0) + glyph.bottom_left;
			s1 = pointwise_multiply(glyph.dimensions, s1) + glyph.bottom_left;
			if(clip_line(clip, &s0, &s1))
			{
				Vector3 start = make_vector3(s0);
				Vector3 end = make_vector3(s1);
				immediate::add_line(start, end, colour);
			}
		}
	}
}

struct Font
{
	Vector2 glyph_dimensions;
	float bearing_left;
	float bearing_right;
	float tracking;
	float leading;
};

static void draw_text(char* text, Vector2 bottom_left, Rect clip_rect, Font* font, Vector4 colour)
{
	if(!text)
	{
		return;
	}

	Vector2 glyph_dimensions = font->glyph_dimensions;
	float advance = font->bearing_left + font->glyph_dimensions.x + font->bearing_right;
	float leading = font->leading;
	float tracking = font->tracking;

	Vector2 pen = bottom_left;
	while(*text)
	{
		char next_char = *text;
		if(next_char == '\n')
		{
			pen.y -= leading;
			pen.x = bottom_left.x;
		}
		else if(next_char != ' ')
		{
			u16 code = ascii_to_16_segment_code(next_char);
			Rect glyph_rect = {pen, glyph_dimensions};
			draw_16_segment_glyph(code, glyph_rect, clip_rect, colour);
			pen.x += advance + tracking;
		}
		else
		{
			pen.x += advance + tracking;
		}
		text += 1;
	}
	immediate::draw();
}

// §3.9 Input...................................................................

enum class UserKey
{
	Space,
	Left,
	Up,
	Right,
	Down,
	Home,
	End,
	Page_Up,
	Page_Down,
	Tilde,
	Enter,
	S,
	V,
	F2,
};

namespace
{
	const int key_count = 14;

	bool keys_pressed[key_count];
	bool old_keys_pressed[key_count];
	// This counts the frames since the last time the key state changed.
	int edge_counts[key_count];
}

static void update_input_states()
{
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
}

static bool key_tapped(UserKey key)
{
	int which = static_cast<int>(key);
	return keys_pressed[which] && edge_counts[which] == 0;
}

static bool key_released(UserKey key)
{
	int which = static_cast<int>(key);
	return !keys_pressed[which] && edge_counts[which] == 0;
}

static bool key_pressed(UserKey key)
{
	int which = static_cast<int>(key);
	return keys_pressed[which];
}

// §3.10 Scroll Panel...........................................................

static const int scroll_panel_lines_cap = 32;
static const int scroll_panel_line_char_limit = 24;

struct ScrollPanel
{
	char lines[scroll_panel_lines_cap][scroll_panel_line_char_limit];
	Vector2 bottom_left;
	Vector2 dimensions;
	Vector2 padding;
	float scroll;
	float line_height;
	int selected;
	int prior_selected;
	int lines_count;
};

static Rect dilate_rect(Rect rect, Vector2 dilation)
{
	Rect result;
	result.bottom_left = rect.bottom_left - dilation;
	result.dimensions = rect.dimensions + 2.0f * dilation;
	return result;
}

static float get_inner_height(ScrollPanel* panel)
{
	return panel->dimensions.y - 2.0f * panel->padding.y;
}

static float get_max_scroll(ScrollPanel* panel)
{
	const float weird_offset = 0.5f;
	float inner_height = get_inner_height(panel);
	float max = (panel->line_height * panel->lines_count) - inner_height + weird_offset;
	max = fmax(max, 0.0f);
	return max;
}

static void set_scroll(ScrollPanel* panel, float scroll)
{
	float max_scroll = get_max_scroll(panel);
	panel->scroll = clamp(scroll, 0.0f, max_scroll);
}

static void scroll_panel(ScrollPanel* panel, float delta, float target)
{
	float distance = abs(delta);
	const float scroll_speed = 4.0f;
	if(distance < scroll_speed)
	{
		set_scroll(panel, target);
	}
	else
	{
		float move = scroll_speed * delta / distance;
		set_scroll(panel, panel->scroll + move);
	}
}

static void set_selection(ScrollPanel* panel, int index)
{
	int selected = mod(index, panel->lines_count);
	if(index == -1 && selected == panel->lines_count - 1)
	{
		float max_scroll = get_max_scroll(panel);
		set_scroll(panel, max_scroll);
		panel->prior_selected = selected;
	}
	else if(selected == 0 && index == panel->lines_count)
	{
		set_scroll(panel, 0.0f);
		panel->prior_selected = selected;
	}
	else
	{
		panel->prior_selected = panel->selected;
	}
	panel->selected = selected;
}

static void scroll_panel_update(ScrollPanel* panel)
{
	int selected = panel->selected;
	int prior_selected = panel->prior_selected;

	if(selected != prior_selected)
	{
		float threshold = 2.0f * panel->line_height;
		float scroll = panel->scroll;

		if(selected < prior_selected)
		{
			float target = (panel->line_height * selected) - threshold;
			float scroll_top = scroll;
			if(target < scroll_top)
			{
				float delta = target - scroll_top;
				scroll_panel(panel, delta, target);
			}
		}
		else
		{
			float target = (panel->line_height * (selected + 1)) + threshold;
			float inner_height = get_inner_height(panel);
			float scroll_bottom = scroll + inner_height;
			if(target > scroll_bottom)
			{
				float delta = target - scroll_bottom;
				float actual_target = target - inner_height;
				scroll_panel(panel, delta, actual_target);
			}
		}
	}
}

static void scroll_panel_handle_input(ScrollPanel* panel)
{
	if(key_tapped(UserKey::Up))
	{
		set_selection(panel, panel->selected - 1);
	}
	if(key_tapped(UserKey::Down))
	{
		set_selection(panel, panel->selected + 1);
	}
	if(key_tapped(UserKey::Home))
	{
		set_selection(panel, panel->lines_count);
	}
	if(key_tapped(UserKey::End))
	{
		set_selection(panel, -1);
	}
	if(key_tapped(UserKey::Page_Up))
	{
		float inner_height = get_inner_height(panel);
		int lines_per_page = inner_height / panel->line_height;
		int selection = MAX(panel->selected - lines_per_page, 0);
		set_scroll(panel, panel->scroll - inner_height);
		set_selection(panel, selection);
		panel->prior_selected = panel->selected;
	}
	if(key_tapped(UserKey::Page_Down))
	{
		float inner_height = get_inner_height(panel);
		int lines_per_page = inner_height / panel->line_height;
		int selection = MIN(panel->selected + lines_per_page, panel->lines_count - 1);
		set_scroll(panel, panel->scroll + inner_height);
		set_selection(panel, selection);
		panel->prior_selected = panel->selected;
	}
}

static Vector4 get_line_colour(ScrollPanel* panel, int index)
{
	if(index == panel->selected)
	{
		return colour_black;
	}
	return colour_white;
}

static Quad rect_to_quad(Rect r)
{
	float left = r.bottom_left.x;
	float right = r.bottom_left.x + r.dimensions.x;
	float bottom = r.bottom_left.y;
	float top = r.bottom_left.y + r.dimensions.y;

	Quad result;
	result.vertices[0] = {left, bottom, 0.0f};
	result.vertices[1] = {right, bottom, 0.0f};
	result.vertices[2] = {right, top, 0.0f};
	result.vertices[3] = {left, top, 0.0f};
	return result;
}

static bool clip_rects(Rect inner, Rect outer, Rect* result)
{
	float i_right = inner.bottom_left.x + inner.dimensions.x;
	float o_right = outer.bottom_left.x + outer.dimensions.x;
	i_right = fmin(i_right, o_right);

	float i_top = inner.bottom_left.y + inner.dimensions.y;
	float o_top = outer.bottom_left.y + outer.dimensions.y;
	i_top = fmin(i_top, o_top);

	float x = fmax(inner.bottom_left.x, outer.bottom_left.x);
	float y = fmax(inner.bottom_left.y, outer.bottom_left.y);
	float width = i_right - x;
	float height = i_top - y;

	if(width <= 0.0f && height <= 0.0f)
	{
		return false;
	}
	else
	{
		result->bottom_left.x = x;
		result->bottom_left.y = y;
		result->dimensions.x = width;
		result->dimensions.y = height;
		return true;
	}
}

static void draw_rect(Rect rect, Vector4 colour)
{
	Quad quad = rect_to_quad(rect);
	immediate::add_quad(&quad, colour);
	immediate::draw();
}

static void draw_rect_transparent(Rect rect, Vector4 colour)
{
	Quad quad = rect_to_quad(rect);
	immediate::add_quad(&quad, colour);
	immediate::set_blend_mode(immediate::BlendMode::Transparent);
	immediate::draw();
}

static void scroll_panel_draw(ScrollPanel* panel, Font* font)
{
	// Draw the background.

	Rect background_rect;
	{
		Vector2 dimensions = panel->dimensions;
		Vector2 position = panel->bottom_left;
		background_rect = {position, dimensions};
		Vector4 colour = {0.0f, 0.0f, 0.0f, 0.5f};
		draw_rect_transparent(background_rect, colour);
	}

	float scroll = panel->scroll;
	float leading = font->leading;
	int selected = panel->selected;

	// Draw the scrollbar.

	const float scrollbar_width = 16.0f;
	{
		float right = background_rect.bottom_left.x + background_rect.dimensions.x;
		float x = right - scrollbar_width;
		float width = scrollbar_width;

		float max_height = background_rect.dimensions.y;
		float top = background_rect.bottom_left.y + max_height;
		float height = max_height / ((leading * panel->lines_count) / max_height);
		height = clamp(height, scrollbar_width, max_height);
		float max_scroll = get_max_scroll(panel);
		float y = top - height - scroll * (max_height - height) / max_scroll;

		Rect rect = {{x, y}, {width, height}};
		draw_rect(rect, colour_white);
	}

	Rect scroll_area = background_rect;
	scroll_area.dimensions.x -= scrollbar_width;
	scroll_area = dilate_rect(scroll_area, -panel->padding);
	float top = scroll_area.bottom_left.y + scroll_area.dimensions.y;

	// Draw the background of the selected line.
	{
		float x = scroll_area.bottom_left.x;
		float y = -(leading * (selected + 1)) + top + scroll;
		Rect rect = {{x, y}, {scroll_area.dimensions.x, leading}};
		if(clip_rects(rect, scroll_area, &rect))
		{
			draw_rect(rect, colour_white);
		}
	}

	// Draw the lines of text themselves.

	int start_index = scroll / leading;
	start_index = MAX(start_index, 0);
	int end_index = ceil((scroll + scroll_area.dimensions.y) / leading);
	end_index = MIN(end_index, panel->lines_count);

	for(int i = start_index; i < end_index; ++i)
	{
		float x = scroll_area.bottom_left.x;
		float y = -(leading * (i + 1)) + top + scroll;
		Vector2 text_position = {x, y};
		Vector4 colour = get_line_colour(panel, i);
		draw_text(panel->lines[i], text_position, scroll_area, font, colour);
	}
}

// §3.11 Tweaker................................................................
//
// This is my version of Jessica Mak's tweaker. It's a tool that lets you set
// simple variables like numbers and booleans while the game is running. It's
// only for testing and doesn't save the variables when changed.

#if !defined(NDEBUG)
#define TWEAKER_ENABLED
#endif

static const int tweaker_map_entries_count = 16;

struct TweakerMap
{
	enum class EntryType
	{
		Int,
		Float,
		Bool,
	};

	struct Entry
	{
		union
		{
			struct
			{
				int* value;
				int step;
				int range_min;
				int range_max;
			} an_int;

			struct
			{
				float* value;
				float step;
				float range_min;
				float range_max;
			} a_float;

			struct
			{
				bool* value;
			} a_bool;
		};
		const char* name;
		EntryType type;
	} entries[tweaker_map_entries_count];
	int count;
};

static bool register_tweaker_bool(TweakerMap* map, const char* name, bool* value, bool initial)
{
	map->entries[map->count].a_bool.value = value;
	map->entries[map->count].name = name;
	map->entries[map->count].type = TweakerMap::EntryType::Bool;
	map->count += 1;
	ASSERT(map->count < tweaker_map_entries_count);
	return initial;
}

static int register_tweaker_int(TweakerMap* map, const char* name, int* value, int initial, int range_min, int range_max)
{
	map->entries[map->count].an_int.value = value;
	map->entries[map->count].an_int.step = 1;
	map->entries[map->count].an_int.range_min = range_min;
	map->entries[map->count].an_int.range_max = range_max;
	map->entries[map->count].name = name;
	map->entries[map->count].type = TweakerMap::EntryType::Int;
	map->count += 1;
	ASSERT(map->count < tweaker_map_entries_count);
	return initial;
}

static float register_tweaker_float(TweakerMap* map, const char* name, float* value, float initial, float range_min, float range_max)
{
	map->entries[map->count].a_float.value = value;
	map->entries[map->count].a_float.step = 0.1f;
	map->entries[map->count].a_float.range_min = range_min;
	map->entries[map->count].a_float.range_max = range_max;
	map->entries[map->count].name = name;
	map->entries[map->count].type = TweakerMap::EntryType::Float;
	map->count += 1;
	ASSERT(map->count < tweaker_map_entries_count);
	return initial;
}

static const int tweaker_line_char_limit = 24;

struct Tweaker
{
	enum class Mode
	{
		Select,
		Adjust_Value,
		Adjust_Step,
	};

	struct
	{
		char value[tweaker_line_char_limit];
		char step[tweaker_line_char_limit];
		char range[tweaker_line_char_limit];
	} readout;

	ScrollPanel scroll_panel;
	Font font;
	Mode mode;
	bool on;
};

static void tweaker_turn_on(Tweaker* tweaker, bool on)
{
#if defined(TWEAKER_ENABLED)
	if(!tweaker->on && on)
	{
		// When the tweaker is turned on, make sure the following is true.
		tweaker->mode = Tweaker::Mode::Select;
	}
	tweaker->on = on;
#endif
}

static void adjust_value(TweakerMap* map, int selected, bool upward)
{
	TweakerMap::Entry* entry = &map->entries[selected];
	switch(entry->type)
	{
		case TweakerMap::EntryType::Bool:
		{
			bool value = *entry->a_bool.value;
			*entry->a_bool.value = !value;
			break;
		}
		case TweakerMap::EntryType::Int:
		{
			int value = *entry->an_int.value;
			if(upward)
			{
				value += entry->an_int.step;
				value = MIN(value, entry->an_int.range_max);
			}
			else
			{
				 value -= entry->an_int.step;
				 value = MAX(value, entry->an_int.range_min);
			}
			*entry->an_int.value = value;
			break;
		}
		case TweakerMap::EntryType::Float:
		{
			float value = *entry->a_float.value;
			if(upward)
			{
				value += entry->a_float.step;
				value = fmin(value, entry->a_float.range_max);
			}
			else
			{
				value -= entry->a_float.step;
				value = fmax(value, entry->a_float.range_min);
			}
			*entry->a_float.value = value;
			break;
		}
	}
}

static void adjust_step(TweakerMap* map, int selected, bool upward)
{
	TweakerMap::Entry* entry = &map->entries[selected];
	switch(entry->type)
	{
		case TweakerMap::EntryType::Bool:
		{
			// These have no step adjustment, so ignore any input.
			break;
		}
		case TweakerMap::EntryType::Int:
		{
			int step = entry->an_int.step;
			if(upward)
			{
				step *= 10;
				step = MIN(step, 1e6);
			}
			else
			{
				step /= 10;
				step = MAX(step, 1);
			}
			entry->an_int.step = step;
			break;
		}
		case TweakerMap::EntryType::Float:
		{
			float step = entry->a_float.step;
			if(upward)
			{
				step *= 10.0f;
				step = fmin(step, 1e6f);
			}
			else
			{
				step /= 10.0f;
				step = fmax(step, -1e-6f);
			}
			entry->a_float.step = step;
			break;
		}
	}
}

static void try_to_enter_adjust_step_mode(Tweaker* tweaker, TweakerMap* map)
{
	int selected = tweaker->scroll_panel.selected;
	TweakerMap::EntryType type = map->entries[selected].type;
	bool is_step_adjustable =
		type == TweakerMap::EntryType::Int ||
		type == TweakerMap::EntryType::Float;
	if(is_step_adjustable)
	{
		tweaker->mode = Tweaker::Mode::Adjust_Step;
	}
}

static void tweaker_handle_input(Tweaker* tweaker, TweakerMap* map)
{
	switch(tweaker->mode)
	{
		case Tweaker::Mode::Select:
		{
			if(key_tapped(UserKey::V))
			{
				tweaker->mode = Tweaker::Mode::Adjust_Value;
			}
			if(key_tapped(UserKey::S))
			{
				try_to_enter_adjust_step_mode(tweaker, map);
			}
			scroll_panel_handle_input(&tweaker->scroll_panel);
			break;
		}
		case Tweaker::Mode::Adjust_Value:
		{
			if(key_tapped(UserKey::Enter) || key_tapped(UserKey::V))
			{
				tweaker->mode = Tweaker::Mode::Select;
			}
			if(key_tapped(UserKey::S))
			{
				try_to_enter_adjust_step_mode(tweaker, map);
			}
			if(key_tapped(UserKey::Up) || key_tapped(UserKey::Right))
			{
				adjust_value(map, tweaker->scroll_panel.selected, true);
			}
			if(key_tapped(UserKey::Down) || key_tapped(UserKey::Left))
			{
				adjust_value(map, tweaker->scroll_panel.selected, false);
			}
			break;
		}
		case Tweaker::Mode::Adjust_Step:
		{
			if(key_tapped(UserKey::S) || key_tapped(UserKey::Enter))
			{
				tweaker->mode = Tweaker::Mode::Select;
			}
			if(key_tapped(UserKey::V))
			{
				tweaker->mode = Tweaker::Mode::Adjust_Value;
			}
			if(key_tapped(UserKey::Up) || key_tapped(UserKey::Right))
			{
				adjust_step(map, tweaker->scroll_panel.selected, true);
			}
			if(key_tapped(UserKey::Down) || key_tapped(UserKey::Left))
			{
				adjust_step(map, tweaker->scroll_panel.selected, false);
			}
			break;
		}
	}
}

static void tweaker_update(Tweaker* tweaker, TweakerMap* map)
{
#if defined(TWEAKER_ENABLED)
	if(!tweaker->on)
	{
		return;
	}

	tweaker_handle_input(tweaker, map);

	// Fill out the left panel with entry names.
	for(int i = 0; i < map->count; ++i)
	{
		copy_string(tweaker->scroll_panel.lines[i], scroll_panel_line_char_limit, map->entries[i].name);
	}

	// Fill the right panel with the readout value and the step amount.

	int selected = tweaker->scroll_panel.selected;

	if(!map->entries[selected].name)
	{
		empty_string(tweaker->readout.value);
		empty_string(tweaker->readout.step);
		empty_string(tweaker->readout.range);
	}
	else
	{
		switch(map->entries[selected].type)
		{
			case TweakerMap::EntryType::Bool:
			{
				bool b = *map->entries[selected].a_bool.value;
				const char* value = bool_to_string(b);
				snprintf(tweaker->readout.value, tweaker_line_char_limit, "[V]alue=%s", value);
				empty_string(tweaker->readout.step);
				empty_string(tweaker->readout.range);
				break;
			}
			case TweakerMap::EntryType::Int:
			{
				int value = *map->entries[selected].an_int.value;
				int step = map->entries[selected].an_int.step;
				int range_min = map->entries[selected].an_int.range_min;
				int range_max = map->entries[selected].an_int.range_max;
				snprintf(tweaker->readout.value, tweaker_line_char_limit, "[V]alue=%d", value);
				snprintf(tweaker->readout.step, tweaker_line_char_limit, "[S]tep=%d", step);
				if(range_min == INT_MIN && range_max == INT_MAX)
				{
					empty_string(tweaker->readout.range);
				}
				else
				{
					snprintf(tweaker->readout.range, tweaker_line_char_limit, "Range=[%d,%d]", range_min, range_max);
				}
				break;
			}
			case TweakerMap::EntryType::Float:
			{
				float value = *map->entries[selected].a_float.value;
				float step = map->entries[selected].a_float.step;
				float range_min = map->entries[selected].a_float.range_min;
				float range_max = map->entries[selected].a_float.range_max;
				snprintf(tweaker->readout.value, tweaker_line_char_limit, "[V]alue=%f", value);
				snprintf(tweaker->readout.step, tweaker_line_char_limit, "[S]tep=%f", step);
				if(range_min == -infinity && range_max == +infinity)
				{
					empty_string(tweaker->readout.range);
				}
				else
				{
					snprintf(tweaker->readout.range, tweaker_line_char_limit, "Range=[%g,%g]", range_min, range_max);
				}
				break;
			}
		}
	}

	// Update the scroll position in the left panel.
	scroll_panel_update(&tweaker->scroll_panel);
#endif
}

static void draw_tweaker(Tweaker* tweaker)
{
#if defined(TWEAKER_ENABLED)
	if(!tweaker->on)
	{
		// peace right out
		return;
	}

	scroll_panel_draw(&tweaker->scroll_panel, &tweaker->font);

	float leading = tweaker->font.leading;

	Rect left_panel;
	{
		Vector2 dimensions = tweaker->scroll_panel.dimensions;
		Vector2 position = tweaker->scroll_panel.bottom_left;
		left_panel = {position, dimensions};
	}

	// Draw the background of the right panel.

	Rect right_panel = left_panel;
	right_panel.bottom_left.x += right_panel.dimensions.x;

	Vector4 colour = {0.0f, 0.0f, 0.0f, 0.5f};
	draw_rect_transparent(right_panel, colour);

	// Draw the readout in the right panel.

	float left = right_panel.bottom_left.x;
	float top = right_panel.bottom_left.y + right_panel.dimensions.y;

	Vector2 text_position = {left, top - leading};
	draw_text(tweaker->readout.value, text_position, right_panel, &tweaker->font, colour_white);
	text_position.y -= leading;
	draw_text(tweaker->readout.step, text_position, right_panel, &tweaker->font, colour_white);
	text_position.y -= leading;
	draw_text(tweaker->readout.range, text_position, right_panel, &tweaker->font, colour_white);

	// Draw indicators for the current mode.

	float right = right_panel.bottom_left.x + right_panel.dimensions.x;
	top = right_panel.bottom_left.y + right_panel.dimensions.y;
	text_position = {right - 16.0f, top - leading};
	char* mode_indicator = const_cast<char*>("<");
	if(tweaker->mode == Tweaker::Mode::Adjust_Value)
	{
		draw_text(mode_indicator, text_position, right_panel, &tweaker->font, colour_white);
	}
	text_position.y -= leading;
	if(tweaker->mode == Tweaker::Mode::Adjust_Step)
	{
		draw_text(mode_indicator, text_position, right_panel, &tweaker->font, colour_white);
	}

	// Draw right panel hints.

	char* hint;
	if(tweaker->mode == Tweaker::Mode::Adjust_Step)
	{
		hint = const_cast<char*>("[Enter] or [S] to\nfinish editing");
	}
	else if(tweaker->mode == Tweaker::Mode::Adjust_Value)
	{
		hint = const_cast<char*>("[Enter] or [V] to\nfinish editing");
	}
	else
	{
		hint = nullptr;
	}

	if(hint)
	{
		float bottom = right_panel.bottom_left.y;
		const float weird_offset = 0.5f;
		text_position = {left, bottom + leading + weird_offset};
		draw_text(hint, text_position, right_panel, &tweaker->font, colour_white);
	}
#endif
}

// The tweaker shouldn't be present in release mode. It isn't needed, but more
// importantly these macros use static initialisation to register its variables
// which causes structures to become non-POD types. All structures in this file
// are assumed to be POD, so many behaviours would become undefined. This isn't
// likely to be any trouble in development but is strictly unacceptable for
// shipping.
//
// Also, the tweaker acts on and monitors variables in a totally non-thread-safe
// way, which is again, unacceptable for release.
#if defined(TWEAKER_ENABLED)

#define TWEAKER_BOOL(name, initial)\
	bool name = register_tweaker_bool(&tweaker_map, #name, &name, initial);

#define TWEAKER_INT(name, initial)\
	int name = register_tweaker_int(&tweaker_map, #name, &name, initial, INT_MIN, INT_MAX);

#define TWEAKER_INT_RANGE(name, initial, min, max)\
	int name = register_tweaker_int(&tweaker_map, #name, &name, initial, min, max);

#define TWEAKER_FLOAT(name, initial)\
	float name = register_tweaker_float(&tweaker_map, #name, &name, initial, -infinity, +infinity);

#define TWEAKER_FLOAT_RANGE(name, initial, min, max)\
	float name = register_tweaker_float(&tweaker_map, #name, &name, initial, min, max);
#else

#define TWEAKER_BOOL(name, initial)\
	bool name;

#define TWEAKER_INT(name, initial)\
	int name;

#define TWEAKER_INT_RANGE(name, initial, min, max)\
	int name;

#define TWEAKER_FLOAT(name, initial)\
	float name;

#define TWEAKER_FLOAT_RANGE(name, initial, min, max)\
	float name;

#endif // defined(TWEAKER_ENABLED)

namespace
{
	Tweaker tweaker;
	TweakerMap tweaker_map;
}

// §3.12 Oscilloscope...........................................................

static const int oscilloscope_channel_samples_count = 32768;
static const int oscilloscope_channels_count = 2;
static const float oscilloscope_hysteresis = 0.01f;
static const int trace_points_count = 256;

struct Oscilloscope
{
	struct Channel
	{
		float samples[oscilloscope_channel_samples_count];
		int samples_index;
		int samples_buffered;
		int trigger_index;
		bool active;
	};

	Channel channels[oscilloscope_channels_count];
	float timebase;
	float holdoff;
	int sample_rate;
	float range;
	float trigger_level;
	bool normalise;
	bool disable_trigger;
	bool trigger_rising_edge;
};

static void oscilloscope_default(Oscilloscope* scope)
{
	scope->sample_rate = 44100;
	scope->holdoff = 500e-9f;
	scope->timebase = 1.0f / 60.0f;
	scope->range = 2.0f;
	scope->trigger_level = 0.0f;
	scope->disable_trigger = false;
	scope->trigger_rising_edge = true;
}

static int get_cyclic_distance(int a, int b, int range)
{
	if(a < b)
	{
		return b - a;
	}
	else
	{
		return b + (range - a);
	}
}

static bool in_cyclic_interval(int x, int first, int second)
{
	if(second > first)
	{
		return x >= first && x <= second;
	}
	else
	{
		return x >= first || x <= second;
	}
}

static void oscilloscope_set_trigger(Oscilloscope::Channel* channel, int index)
{
	int prior = channel->trigger_index;
	int distance = get_cyclic_distance(prior, index, oscilloscope_channel_samples_count);
	ASSERT(can_use_bitwise_and_to_cycle(oscilloscope_channel_samples_count));
	int mask = oscilloscope_channel_samples_count - 1;
	int tail = (channel->samples_index + channel->samples_buffered) & mask;
	if(in_cyclic_interval(tail, prior, index))
	{
		channel->trigger_index = tail;
		channel->samples_buffered = 0;
	}
	else
	{
		channel->trigger_index = index;
		channel->samples_buffered -= distance;
	}
}

static void oscilloscope_detect_trigger(Oscilloscope* scope, int channel_index)
{
	Oscilloscope::Channel* channel = &scope->channels[channel_index];

	float trigger_level = scope->trigger_level;
	bool disable_trigger = scope->disable_trigger;
	bool rising = scope->trigger_rising_edge;

	ASSERT(can_use_bitwise_and_to_cycle(oscilloscope_channel_samples_count));
	int samples_mask = oscilloscope_channel_samples_count - 1;

	int samples_to_wait = (scope->timebase + scope->holdoff) * scope->sample_rate;
	int start = (channel->trigger_index + samples_to_wait) & samples_mask;
	if(disable_trigger)
	{
		oscilloscope_set_trigger(channel, start);
		return;
	}

	float first;
	if(rising)
	{
		first = trigger_level - (oscilloscope_hysteresis * scope->range);
	}
	else
	{
		first = trigger_level + (oscilloscope_hysteresis * scope->range);
	}
	float second = trigger_level;

	bool first_crossed = false;
	int i = start;
	float prior;
	float level = channel->samples[i];
	do
	{
		prior = level;
		i = (i + 1) & samples_mask;
		level = channel->samples[i];
		if(prior <= first && level >= first)
		{
			first_crossed = true;
		}
		if(first_crossed && prior <= second && level >= second)
		{
			oscilloscope_set_trigger(channel, i);
			return;
		}
	} while(i != start);

	// If no trigger is detected, act the same as when triggering is disabled.
	oscilloscope_set_trigger(channel, start);
}

static void oscilloscope_sample_data(Oscilloscope* scope, int channel_index, float* samples, int samples_count, int offset, int stride)
{
	Oscilloscope::Channel* channel = &scope->channels[channel_index];
	if(channel->samples_buffered >= oscilloscope_channel_samples_count)
	{
		LOG_DEBUG("Oscilloscope buffer overflowed!");
		return;
	}

	ASSERT(can_use_bitwise_and_to_cycle(oscilloscope_channel_samples_count));
	int o = channel->samples_index;
	for(int i = offset; i < samples_count; i += stride)
	{
		channel->samples[o] = samples[i];
		o = (o + 1) & (oscilloscope_channel_samples_count - 1);
	}
	channel->samples_index = o;

	channel->samples_buffered += samples_count / stride;
}

struct Trace
{
	float points[trace_points_count];
};

static void normalise_trace(Trace* trace)
{
	float min = +infinity;
	float max = -infinity;
	for(int i = 0; i < trace_points_count; ++i)
	{
		float point = trace->points[i];
		min = fmin(min, point);
		max = fmax(max, point);
	}
	for(int i = 0; i < trace_points_count; ++i)
	{
		float x = trace->points[i];
		x = (x - min) / (max - min);
		trace->points[i] = 2.0f * x - 1.0f;
	}
}

static void trace_oscilloscope_channel(Trace* trace, Oscilloscope* scope, int channel_index)
{
	oscilloscope_detect_trigger(scope, channel_index);
	ASSERT(can_use_bitwise_and_to_cycle(oscilloscope_channel_samples_count));
	Oscilloscope::Channel* channel = &scope->channels[channel_index];
	int window = scope->timebase * scope->sample_rate;
	for(int i = 0; i < trace_points_count; ++i)
	{
		int step = window * i / static_cast<float>(trace_points_count);
		int o = (channel->trigger_index + step) & (oscilloscope_channel_samples_count - 1);
		trace->points[i] = channel->samples[o];
	}
	if(scope->normalise)
	{
		normalise_trace(trace);
	}
}

static void draw_trace(Trace* trace, float x, float y, float width, float height)
{
	Vector4 colour = colour_white;
	float scale_y = height / 2.0f;
	for(int i = 0; i < trace_points_count - 1; ++i)
	{
		float x0 = width * i / static_cast<float>(trace_points_count) + x;
		float x1 = width * (i + 1) / static_cast<float>(trace_points_count) + x;
		float y0 = scale_y * trace->points[i] + y;
		float y1 = scale_y * trace->points[i + 1] + y;
		Vector3 p0 = {x0, y0, 0.0f};
		Vector3 p1 = {x1, y1, 0.0f};
		immediate::add_line(p0, p1, colour);
	}
	immediate::draw();
}

// §3.13 Profile Inspector......................................................

namespace profile {

struct Record
{
	const char* name;
	u64 ticks;
	int calls;
	int indent;
};

typedef volatile u32 SpinLock;

void spin_lock_acquire(SpinLock* lock);
void spin_lock_release(SpinLock* lock);

static const int thread_history_thread_count = 2;
static const int thread_history_book_count = 128;

struct ThreadHistory
{
	Record* records[thread_history_thread_count][thread_history_book_count];
	int records_count[thread_history_thread_count][thread_history_book_count];
	int records_capacity[thread_history_thread_count][thread_history_book_count];
	int indices[thread_history_thread_count];
	SpinLock lock;
};

struct Inspector
{
	ScrollPanel scroll_panel;
	Font font;
	ThreadHistory history;
	bool on;
	bool halt_collection;
	u8 animation_counter;
};

static void inspector_turn_on(Inspector* inspector, bool on)
{
#if defined(PROFILE_ENABLED)
	if(!inspector->on && on)
	{
		inspector->halt_collection = false;
	}
	inspector->on = on;
#endif
}

static void inspector_handle_input(Inspector* inspector)
{
	if(key_tapped(UserKey::Enter))
	{
		inspector->halt_collection = !inspector->halt_collection;
	}
}

static void function_name_from_signature(char* name, int name_cap, const char* signature)
{
	int parens = find_char(signature, '(');
	if(parens != -1)
	{
		int space = find_last_char(signature, ' ', parens);
		int colon = find_last_char(signature, ':', parens);
		int start = MAX(space, colon);
		if(start != -1)
		{
			int name_limit = MIN(parens - start, name_cap);
			copy_string(name, name_limit, signature + start + 1);
		}
		else
		{
			copy_string(name, name_cap, signature);
		}
	}
	else
	{
		copy_string(name, name_cap, signature);
	}
}

static void inspector_fill_lines(Inspector* inspector)
{
	ScrollPanel* panel = &inspector->scroll_panel;
	panel->lines_count = 0;

	ThreadHistory* history = &inspector->history;
	spin_lock_acquire(&history->lock);

	for(int i = 0; i < thread_history_thread_count; ++i)
	{
		int book = mod(history->indices[i] - 1, thread_history_book_count);
		Record* records = history->records[i][book];
		int records_count = history->records_count[i][book];
		for(int j = 0; j < records_count; ++j)
		{
			char* line = panel->lines[panel->lines_count];
			panel->lines_count = MIN(panel->lines_count + 1, scroll_panel_lines_cap - 1);
			ASSERT(panel->lines_count + 1 < scroll_panel_lines_cap);

			Record* record = &records[j];
			int indent = record->indent;

			for(int k = 0; k < indent; ++k)
			{
				line[k] = ' ';
			}

			const int name_cap = 32;
			char name[name_cap];
			function_name_from_signature(name, name_cap, record->name);

			int char_limit = scroll_panel_line_char_limit - indent;
			int name_limit = 12 - indent;
			int milliticks = record->ticks / 1000;
			snprintf(line + indent, char_limit, "%-*.*s %6d %3d", name_limit, name_limit, name, milliticks, record->calls);
		}
	}

	spin_lock_release(&history->lock);
}

static void inspector_update(Inspector* inspector)
{
#if defined(PROFILE_ENABLED)
	scroll_panel_handle_input(&inspector->scroll_panel);
	inspector_handle_input(inspector);
	scroll_panel_update(&inspector->scroll_panel);
	inspector_fill_lines(inspector);
#endif
}

const int qualitative_palette_cap = 30;

static const u32 qualitative_palette[qualitative_palette_cap] =
{
	0x573bce,
	0x8600f5,
	0xcf00dd,
	0xf30053,
	0xff3a00,
	0xfcaf29,
	0x2000ec,
	0x348ac7,
	0x1b617a,
	0x008486,
	0x6caf7f,
	0xfd5a35,
	0x105283,
	0x035774,
	0x007f89,
	0xaec9a7,
	0xf3d489,
	0xffb058,
	0x00325b,
	0x005171,
	0x007c8a,
	0xd4d6bd,
	0xeee9bb,
	0xffe06a,
	0x728398,
	0x5c8aae,
	0x34bac5,
	0xb5d568,
	0xe0c769,
	0xe16a69,
};

static u64 compute_global_max_ticks(ThreadHistory* history)
{
	u64 max = 0;
	for(int i = 0; i < thread_history_thread_count; ++i)
	{
		for(int j = 0; j < thread_history_book_count; ++j)
		{
			Record* records = history->records[i][j];
			if(!records)
			{
				continue;
			}
			int records_count = history->records_count[i][j];
			for(int k = 0; k < records_count; ++k)
			{
				u64 ticks = records[k].ticks;
				max = MAX(max, ticks);
			}
		}
	}
	return max;
}

struct Channel
{
	const char* name;
	u32 colour;
};

static const int channels_cap = 32;

static int find_channel(Channel* channels, int channels_count, const char* name)
{
	for(int i = 0; i < channels_count; ++i)
	{
		if(channels[i].name == name)
		{
			return i;
		}
	}
	return -1;
}

static int find_selected(Channel* channels, int channels_count, ThreadHistory* history, int selected)
{
	for(int i = 0; i < thread_history_thread_count; ++i)
	{
		int book = mod(history->indices[i] - 1, thread_history_book_count);
		Record* records = history->records[i][book];
		if(!records)
		{
			continue;
		}
		int records_count = history->records_count[i][book];
		if(selected < records_count)
		{
			const char* name = records[selected].name;
			return find_channel(channels, channels_count, name);
		}
		else
		{
			selected -= records_count;
		}
	}
	return -1;
}

static void assign_channels(ThreadHistory* history, Channel* channels, int* result_count)
{
	int channels_count = 0;
	int colour_count = 0;
	for(int i = 0; i < thread_history_thread_count; ++i)
	{
		for(int j = 0; j < thread_history_book_count; ++j)
		{
			Record* records = history->records[i][j];
			if(!records)
			{
				continue;
			}
			int records_count = history->records_count[i][j];
			for(int k = 0; k < records_count; ++k)
			{
				const char* name = records[k].name;
				int found_index = find_channel(channels, channels_count, name);
				if(found_index == -1)
				{
					channels[channels_count].name = name;
					channels[channels_count].colour = qualitative_palette[colour_count];
					channels_count += 1;
					ASSERT(channels_count < channels_cap);
					colour_count = (colour_count + 1) % qualitative_palette_cap;
				}
			}
		}
	}
	*result_count = channels_count;
}

static void inspector_draw(Inspector* inspector)
{
#if defined(PROFILE_ENABLED)
	if(!inspector->on)
	{
		return;
	}

	ScrollPanel* panel = &inspector->scroll_panel;

	Vector4 bar_colour = {0.0f, 0.0f, 0.0f, 0.6f};
	float bar_height = 30.0f;

	Rect rect;
	rect.bottom_left.x = panel->bottom_left.x;
	rect.bottom_left.y = panel->bottom_left.y + panel->dimensions.y;
	rect.dimensions.x = panel->dimensions.x;
	rect.dimensions.y = bar_height;
	draw_rect_transparent(rect, bar_colour);

	const char* titles = "function     mticks calls";
	Vector2 title_position = rect.bottom_left;
	title_position.y += 4.0f;
	draw_text(const_cast<char*>(titles), title_position, rect, &inspector->font, colour_white);

	scroll_panel_draw(&inspector->scroll_panel, &inspector->font);

	rect.bottom_left.y = panel->bottom_left.y - bar_height;
	draw_rect_transparent(rect, bar_colour);

	const char* hint = "[Enter] to pause";
	title_position = rect.bottom_left;
	title_position.y += 4.0f;
	draw_text(const_cast<char*>(hint), title_position, rect, &inspector->font, colour_white);

	Vector4 chart_bar_colour = {0.0f, 0.0f, 0.0f, 0.5f};
	rect.bottom_left.y -= 120.0f;
	rect.dimensions.y = 110.0f;
	draw_rect_transparent(rect, chart_bar_colour);

	ThreadHistory* history = &inspector->history;

	// Find the maximum cycles of all records so that the drawing of each trace
	// can be scaled to fit the chart height.
	u64 max_ticks = compute_global_max_ticks(history);

	Channel channels[channels_cap];
	int channels_count;
	assign_channels(history, channels, &channels_count);
	struct
	{
		Vector2 p0;
		int index;
	} traces[channels_cap] = {};
	for(int i = 0; i < channels_count; ++i)
	{
		// Assign any invalid index so that checks against valid indices will
		// for sure be false.
		traces[i].index = -1;
	}

	// Find selected channel.
	int selected = inspector->scroll_panel.selected;
	int selected_channel = find_selected(channels, channels_count, history, selected);
	ASSERT(selected_channel != -1);

	// Animate the colour of the trace for the selected channel.
	float theta = 8.0f * tau * inspector->animation_counter / 255.0f;
	inspector->animation_counter += 1;
	float phase = 0.5f * sin(theta) + 0.5f;
	u32 selected_colour = r_to_u32(phase);

	// Draw the traces.
	for(int i = 0; i < thread_history_thread_count; ++i)
	{
		int tail = history->indices[i];
		int wrap = thread_history_book_count;
		for(int head = mod(tail - 1, wrap), prior_head = tail; head != tail; prior_head = head, head = mod(head - 1, wrap))
		{
			Record* records = history->records[i][head];
			if(!records)
			{
				// If there's fewer than books than thread_history_book_count, the
				// end of the line may come sooner than encountering the tail.
				break;
			}

			float sibling_adjust = 0.0f;
			int prior_indent = -1;
			int records_count = history->records_count[i][head];
			for(int j = 0; j < records_count; ++j)
			{
				Record* record = &records[j];

				Vector2 p;
				p.x = head / static_cast<float>(thread_history_book_count);
				p.y = record->ticks / static_cast<float>(max_ticks);

				if(record->indent == prior_indent)
				{
					p.y += sibling_adjust;
					sibling_adjust = p.y;
				}
				else
				{
					prior_indent = record->indent;
					sibling_adjust = 0.0f;
				}

				Vector2 p1 = pointwise_multiply(p, rect.dimensions) + rect.bottom_left;
				int found_index = find_channel(channels, channels_count, record->name);
				ASSERT(found_index != -1);
				if(traces[found_index].index == prior_head)
				{
					Vector2 p0 = traces[found_index].p0;
					Vector3 end = make_vector3(p0);
					Vector3 start = make_vector3(p1);
					if(prior_head == 0)
					{
						end.x = rect.bottom_left.x + rect.dimensions.x;
					}
					u32 colour_u32;
					if(found_index == selected_channel)
					{
						colour_u32 = selected_colour;
					}
					else
					{
						colour_u32 = channels[found_index].colour;
					}
					Vector4 colour = u32_to_rgba(colour_u32);
					immediate::add_line(start, end, colour);
				}
				traces[found_index].index = head;
				traces[found_index].p0 = p1;
			}
		}
	}
	immediate::draw();

	// Draw markers where the traces begin and end.
	Vector4 light_grey = {0.8f, 0.8f, 0.8f, 1.0f};
	Vector4 marker_colours[thread_history_thread_count] = {colour_white, light_grey};
	float bottom = rect.bottom_left.y;
	float top = bottom + rect.dimensions.y;
	for(int i = 0; i < thread_history_thread_count; ++i)
	{
		int tail = history->indices[i];
		float x = tail / static_cast<float>(thread_history_book_count);
		x = x * rect.dimensions.x + rect.bottom_left.x;
		Vector3 start = {x, bottom, 0.0f};
		Vector3 end = {x, top, 0.0f};
		Vector4 colour = marker_colours[i];
		immediate::add_line(start, end, colour);
	}
	immediate::draw();
#endif
}

} // namespace profile

// Jittered Grid................................................................

static void draw_particles(Vector2* points, int points_count, Vector3 bottom_left)
{
	const float radius = 0.02f;
	Vector3 offsets[4] =
	{
		{-radius, -radius, 0.0f},
		{+radius, -radius, 0.0f},
		{+radius, +radius, 0.0f},
		{-radius, +radius, 0.0f},
	};
	for(int i = 0; i < points_count; ++i)
	{
		Quad quad;
		for(int j = 0; j < 4; ++j)
		{
			quad.vertices[j] = bottom_left + make_vector3(points[i]) + offsets[j];
		}
		immediate::add_quad(&quad, colour_white);
	}
	immediate::draw();
}

static Vector3 reflect(Vector3 v, Vector3 n)
{
	return v - (2.0f * dot(v, n) / squared_length(n)) * n;
}

static Vector3 pick_point_in_triangle(Vector3 u, Vector3 v, arandom::Sequence* randomness)
{
	float a0 = arandom::float_range(randomness, 0.0f, 1.0f);
	float a1 = arandom::float_range(randomness, 0.0f, 1.0f);
	Vector3 parallelogram_point = (a0 * u) + (a1 * v);
	Vector3 triangle_point = parallelogram_point;
	if(a0 + a1 > 1.0f)
	{
		// If it's in the half of the parallelogram outside the triangle
		// reflect it back into the triangle using the center plane.
		Vector3 normal = -(u + v) / 2.0f;
		triangle_point = reflect(parallelogram_point, normal);
	}
	return triangle_point;
}

static Vector3* create_particles(int* result_count, arandom::Sequence* randomness)
{
	const int width = 16;
	const int height = 16;
	const float spacing = 0.4f;
	const float altitude = sqrt(3.0f) / 2.0f;
	int points_count = width * height;
	Vector3* points = ALLOCATE(Vector3, points_count);
	Vector3 uv[3] =
	{
		{spacing * 0.5f, spacing * -altitude, 0.0f},
		{spacing, 0.0f, 0.0f},
		{spacing * 0.5f, spacing * +altitude, 0.0f},
	};
	for(int i = 0; i < height; ++i)
	{
		int row_flip = i % 2;
		float half_base = 0.5f * row_flip;
		for(int j = 0; j < width; ++j)
		{
			float x = spacing * (j + half_base);
			float y = spacing * altitude * i;
			float z = 0.0f;
			Vector3 point = {x, y, z};
			Vector3 u = uv[row_flip];
			Vector3 v = uv[row_flip + 1];
			Vector3 nudge = pick_point_in_triangle(u, v, randomness);
			points[width * i + j] = point + nudge;
		}
	}
	*result_count = points_count;
	return points;
}

// Blue Noise using Mitchell's Best Candidate...................................

static float toroidal_distance_squared(Vector2 v0, Vector2 v1, float circumference)
{
	float dx = abs(v1.x - v0.x);
	float dy = abs(v1.y - v0.y);
	if(dx > circumference / 2.0f)
	{
		dx = circumference - dx;
	}
	if(dy > circumference / 2.0f)
	{
		dy = circumference - dy;
	}
	return (dx * dx) + (dy * dy);
}

static const int spash_cell_indices_cap = 6;

// Spatial Hash Cell
struct SpashCell
{
	int indices[spash_cell_indices_cap];
	int count;
};

static Vector2* create_blue_noise(int samples, float side, arandom::Sequence* randomness)
{
	const float factor = 1.0f;

	int points_cap = samples;
	int points_count = 0;
	Vector2* points = ALLOCATE(Vector2, points_cap);

	int grid_side = sqrt(samples) / 3;
	grid_side *= grid_side;
	SpashCell* grid = ALLOCATE(SpashCell, grid_side * grid_side);

	for(int i = 0; i < points_cap; ++i, ++points_count)
	{
		// Generate some number of candidate points and choose the sample
		// furthest from all the existing points.
		int candidates = factor * points_count + 1;
		Vector2 best_candidate = vector2_zero;
		float best_distance = 0.0f;
		int best_cell_x = 0;
		int best_cell_y = 0;
		for(int j = 0; j < candidates; ++j)
		{
			float x = arandom::float_range(randomness, 0.0f, side);
			float y = arandom::float_range(randomness, 0.0f, side);
			Vector2 candidate = {x, y};
			// Search the grid cell where the candidate is for potentially
			// close points.
			int gx = (grid_side - 1) * (x / side);
			int gy = (grid_side - 1) * (y / side);
			float min = +infinity;
			SpashCell* cell = &grid[grid_side * gy + gx];
			for(int k = 0; k < cell->count; ++k)
			{
				int index = cell->indices[k];
				Vector2 close = points[index];
				float d = toroidal_distance_squared(candidate, close, side);
				min = fmin(min, d);
			}
			// If the closest point to this candidate is further than any prior
			// candidate, then it's the new best.
			if(min > best_distance)
			{
				best_distance = min;
				best_candidate = candidate;
				best_cell_x = gx;
				best_cell_y = gy;
			}
		}
		// Add the picked index to the containing grid cell and its neighbors.
		for(int j = -1; j <= 1; ++j)
		{
			for(int k = -1; k <= 1; ++k)
			{
				int x = mod(best_cell_x + k, grid_side);
				int y = mod(best_cell_y + j, grid_side);
				SpashCell* cell = &grid[grid_side * y + x];
				cell->indices[cell->count] = i;
				cell->count = MIN(cell->count + 1, spash_cell_indices_cap - 1);
				ASSERT(cell->count < spash_cell_indices_cap - 1);
			}
		}
		points[i] = best_candidate;
	}

	SAFE_DEALLOCATE(grid);

	return points;
}

// Marching Squares.............................................................

struct Floatmap
{
	float* values;
	int columns;
	int rows;
};

namespace marching_squares {

static const u8 edge_table[16] =
{
	0x00, 0x09, 0x03, 0x0a,
	0x06, 0x0f, 0x05, 0x0c,
	0x0c, 0x05, 0x0f, 0x06,
	0x0a, 0x03, 0x09, 0x00
};

static const s8 index_table[16][4] =
{
	{-1, -1, -1, -1},
	{3, 0, -1, -1},
	{0, 1, -1, -1},
	{3, 1, -1, -1},
	{1, 2, -1, -1},
	{0, 1, 2, 3},
	{0, 2, -1, -1},
	{2, 3, -1, -1},
	{2, 3, -1, -1},
	{0, 2, -1, -1},
	{3, 0, 1, 2},
	{1, 2, -1, -1},
	{3, 1, -1, -1},
	{0, 1, -1, -1},
	{3, 0, -1, -1},
	{-1, -1, -1, -1},
};

static Vector2 interpolate_vertex(float isovalue, Vector2 p0, Vector2 p1, float i0, float i1)
{
	if(almost_equals(i0, i1))
	{
		return p0;
	}
	float t = unlerp(i0, i1, isovalue);
	return lerp(p0, p1, t);
}

void delineate(Floatmap* map, float isovalue, Vector2 scale, LineSegment** result, int* result_count)
{
	int columns = map->columns;
	int rows = map->rows;

	LineSegment* lines = nullptr;
	int lines_count = 0;
	int lines_capacity = 0;

	for(int i = 0; i < rows - 1; ++i)
	{
		for(int j = 0; j < columns - 1; ++j)
		{
			float x[4];
			x[0] = map->values[columns * (i    ) + (j    )];
			x[1] = map->values[columns * (i    ) + (j + 1)];
			x[2] = map->values[columns * (i + 1) + (j + 1)];
			x[3] = map->values[columns * (i + 1) + (j    )];

			Vector2 p[4];
			p[0] = {static_cast<float>(j    ), static_cast<float>(i    )};
			p[1] = {static_cast<float>(j + 1), static_cast<float>(i    )};
			p[2] = {static_cast<float>(j + 1), static_cast<float>(i + 1)};
			p[3] = {static_cast<float>(j    ), static_cast<float>(i + 1)};

			int square_index = 0;
			if(x[0] < isovalue) square_index |= 1;
			if(x[1] < isovalue) square_index |= 2;
			if(x[2] < isovalue) square_index |= 4;
			if(x[3] < isovalue) square_index |= 8;

			u8 edge = edge_table[square_index];
			if(edge != 0)
			{
				ENSURE_ARRAY_SIZE(lines, 2);

				Vector2 vertices[4];
				if(edge & 1) vertices[0] = interpolate_vertex(isovalue, p[0], p[1], x[0], x[1]);
				if(edge & 2) vertices[1] = interpolate_vertex(isovalue, p[1], p[2], x[1], x[2]);
				if(edge & 4) vertices[2] = interpolate_vertex(isovalue, p[2], p[3], x[2], x[3]);
				if(edge & 8) vertices[3] = interpolate_vertex(isovalue, p[3], p[0], x[3], x[0]);

				for(int k = 0; k < 4 && index_table[square_index][k] != -1; k += 2)
				{
					Vector2 v0 = vertices[index_table[square_index][k]];
					Vector2 v1 = vertices[index_table[square_index][k + 1]];
					v0 = pointwise_multiply(v0, scale);
					v1 = pointwise_multiply(v1, scale);
					lines[lines_count].vertices[0] = v0;
					lines[lines_count].vertices[1] = v1;
					lines_count += 1;
				}
			}
		}
	}

	*result = lines;
	*result_count = lines_count;
}

void draw_metaballs(Floatmap* map, arandom::Sequence* randomness)
{
	int columns = map->columns;
	int rows = map->rows;

	Vector2 grid_min = vector2_zero;
	Vector2 grid_max;
	grid_max.x = columns - 1;
	grid_max.y = rows - 1;

	const int metaballs = 7;

	for(int i = 0; i < metaballs; ++i)
	{
		Vector2 center;
		center.x = arandom::float_range(randomness, 0.0f, columns - 1);
		center.y = arandom::float_range(randomness, 0.0f, rows - 1);

		float radius = arandom::float_range(randomness, 3.0f, 12.0f);
		float rs = radius * radius;
		Vector2 extents = {radius, radius};

		Vector2 min = center - extents;
		Vector2 max = center + extents;
		min = max2(min, grid_min);
		max = min2(max, grid_max);

		int lx = min.x;
		int ly = min.y;
		int hx = max.x;
		int hy = max.y;

		for(int j = ly; j <= hy; ++j)
		{
			for(int k = lx; k <= hx; ++k)
			{
				Vector2 p;
				p.x = k + 0.5f;
				p.y = j + 0.5f;
				Vector2 v = p - center;
				float d = squared_length(v);
				if(d <= rs)
				{
					float value = (rs - d) / rs;
					int index = columns * j + k;
					map->values[index] = fmin(map->values[index] + value, 1.0f);
				}
			}
		}
	}
}

} // namespace marching_squares

// Spline.......................................................................

static void segment_spline(Vector3* points, int points_count, Vector3* result, int segments)
{
	// Pad the ends of the polyline to use as terminal control points.
	points[0] = (points[1] - points[2]) + points[1];
	int end = points_count - 2;
	points[end + 1] = (points[end] - points[end - 1]) + points[end];

	auto blend = [](Vector3 p0, Vector3 p1, float t, float ta, float tb)
	{
		float u = (ta - t) / (ta - tb);
		float v = (t - tb) / (ta - tb);
		return (u * p0) + (v * p1);
	};

	for(int i = 0; i < points_count - 3; ++i)
	{
		// control points
		Vector3 p[4];
		p[0] = points[i];
		p[1] = points[i + 1];
		p[2] = points[i + 2];
		p[3] = points[i + 3];

		// knot parameters
		float t[4];
		t[0] = 0.0f;
		t[1] = sqrt(distance(p[0], p[1])) + t[0];
		t[2] = sqrt(distance(p[1], p[2])) + t[1];
		t[3] = sqrt(distance(p[2], p[3])) + t[2];

		for(int j = 0; j <= segments; ++j)
		{
			float q = j / static_cast<float>(segments);
			float tj = lerp(t[1], t[2], q);
			Vector3 a0 = blend(p[0], p[1], tj, t[1], t[0]);
			Vector3 a1 = blend(p[1], p[2], tj, t[2], t[1]);
			Vector3 a2 = blend(p[2], p[3], tj, t[3], t[2]);
			Vector3 b0 = blend(a0, a1, tj, t[2], t[0]);
			Vector3 b1 = blend(a1, a2, tj, t[3], t[1]);
			Vector3 c  = blend(b0, b1, tj, t[2], t[1]);
			result[(segments + 1) * i + j] = c;
		}
	}
}

static Vector3 random_point_in_box(AABB bounds, arandom::Sequence* randomness)
{
	Vector3 result;
	result.x = arandom::float_range(randomness, bounds.min.x, bounds.max.x);
	result.y = arandom::float_range(randomness, bounds.min.y, bounds.max.y);
	result.z = arandom::float_range(randomness, bounds.min.z, bounds.max.z);
	return result;
}

static void generate_random_spline(arandom::Sequence* randomness, Vector3** result, int* result_count)
{
	// Make a random polyline.
	const int polyline_points_count = 5;
	const int polyline_cap = polyline_points_count + 2;
	AABB bounds = {{0.0f, 0.0f, 0.0f}, {24.0f, 24.0f, 24.0f}};
	Vector3* polyline = ALLOCATE(Vector3, polyline_cap);
	for(int i = 1; i <= polyline_points_count; ++i)
	{
		polyline[i] = random_point_in_box(bounds, randomness);
	}

	// Interpolate that into a spline and discard the polyline.
	int segments = 10;
	int spline_cap = (polyline_cap - 3) * (segments + 1);
	Vector3* spline = ALLOCATE(Vector3, spline_cap);
	segment_spline(polyline, polyline_cap, spline, segments);
	DEALLOCATE(polyline);

	*result = spline;
	*result_count = spline_cap;
}

// Voxmap.......................................................................

struct Voxmap
{
	u8* voxels;
	int columns;
	int rows;
	int slices;
};

static AABB get_bounds(Voxmap* map)
{
	AABB result;
	result.min = vector3_zero;
	result.max.x = map->columns;
	result.max.y = map->rows;
	result.max.z = map->slices;
	return result;
}

static Vector3 max_rotated_extent(Vector3 v, Matrix4 m)
{
	Vector3 result;
	result.x = (abs(m[0]) * v.x) + (abs(m[1]) * v.y) + (abs(m[2])  * v.z);
	result.y = (abs(m[4]) * v.x) + (abs(m[5]) * v.y) + (abs(m[6])  * v.z);
	result.z = (abs(m[8]) * v.x) + (abs(m[9]) * v.y) + (abs(m[10]) * v.z);
	return result;
}

static AABB bounds_after_transform(AABB box, Matrix4 transform)
{
	AABB result;

	Vector3 center = 0.5f * (box.max + box.min);
	Vector3 extents = 0.5f * (box.max - box.min);

	center = transform * center;
	extents = max_rotated_extent(extents, transform);

	result.min = center - extents;
	result.max = center + extents;

	return result;
}

static void draw_voxmap(Voxmap* canvas, Voxmap* brush, Matrix4 transform)
{
	AABB brush_bounds = get_bounds(brush);
	AABB canvas_bounds = get_bounds(canvas);
	canvas_bounds.max -= vector3_one;

	AABB transformed_bounds = bounds_after_transform(brush_bounds, transform);
	AABB draw_bounds = aabb_clip(canvas_bounds, transformed_bounds);

	int min_x = round(draw_bounds.min.x);
	int min_y = round(draw_bounds.min.y);
	int min_z = round(draw_bounds.min.z);
	int max_x = round(draw_bounds.max.x);
	int max_y = round(draw_bounds.max.y);
	int max_z = round(draw_bounds.max.z);

	Matrix4 inverse = inverse_transform(transform);

	for(int i = min_z; i <= max_z; ++i)
	{
		for(int j = min_y; j <= max_y; ++j)
		{
			for(int k = min_x; k <= max_x; ++k)
			{
				Vector3 canvas_voxel;
				canvas_voxel.x = k;
				canvas_voxel.y = j;
				canvas_voxel.z = i;
				Vector3 brush_voxel = inverse * canvas_voxel;
				int u = round(brush_voxel.x);
				int v = round(brush_voxel.y);
				int w = round(brush_voxel.z);

				if(
					u >= 0 && u < brush->columns &&
					v >= 0 && v < brush->rows &&
					w >= 0 && w < brush->slices)
				{
					int bc = brush->columns;
					int br = brush->rows;
					int brush_index = (bc * br * w) + (bc * v) + u;
					int cc = canvas->columns;
					int cr = canvas->rows;
					int canvas_index = (cc * cr * i) + (cc * j) + k;
					u8 paint = brush->voxels[brush_index];
					canvas->voxels[canvas_index] = MIN(canvas->voxels[canvas_index] + paint, 0xff);
				}
			}
		}
	}
}

static void draw_checkers(Voxmap* map)
{
	int columns = map->columns;
	int rows = map->rows;
	int slices = map->slices;
	for(int i = 0; i < slices; ++i)
	{
		for(int j = 0; j < rows; ++j)
		{
			for(int k = ~(i | j) & 1; k < columns; k += 2)
			{
				int index = (rows * columns * i) + (columns * j) + k;
				map->voxels[index] = 0xff;
			}
		}
	}
}

void draw_metaballs(Voxmap* map, arandom::Sequence* randomness)
{
	int rc = map->rows * map->columns;
	int c = map->columns;

	AABB bounds;
	bounds.min = vector3_zero;
	bounds.max.x = map->columns - 1;
	bounds.max.y = map->rows - 1;
	bounds.max.z = map->slices - 1;

	const int metaballs = 7;
	for(int i = 0; i < metaballs; ++i)
	{
		Vector3 center;
		center.x = arandom::float_range(randomness, bounds.min.x, bounds.max.x);
		center.y = arandom::float_range(randomness, bounds.min.y, bounds.max.y);
		center.z = arandom::float_range(randomness, bounds.min.z, bounds.max.z);

		float radius = arandom::float_range(randomness, 3.0f, 12.0f);

		Vector3 extents = {radius, radius, radius};
		AABB sphere_bounds = aabb_from_ellipsoid(center, extents);
		sphere_bounds.min = max3(sphere_bounds.min, bounds.min);
		sphere_bounds.max = min3(sphere_bounds.max, bounds.max);

		int lx = sphere_bounds.min.x;
		int ly = sphere_bounds.min.y;
		int lz = sphere_bounds.min.z;
		int hx = sphere_bounds.max.x;
		int hy = sphere_bounds.max.y;
		int hz = sphere_bounds.max.z;

		float rs = radius * radius;

		for(int j = lz; j <= hz; ++j)
		{
			for(int k = ly; k <= hy; ++k)
			{
				for(int m = lx; m <= hx; ++m)
				{
					Vector3 p;
					p.x = m;
					p.y = k;
					p.z = j;
					Vector3 v = p - center;
					float d = squared_length(v);
					if(d < rs)
					{
						float value = (rs - d) / rs;
						int index = rc * j + c * k + m;
						map->voxels[index] = fmin(map->voxels[index] + value, 1.0f);
					}
				}
			}
		}
	}
}

// Marching Cubes...............................................................

namespace marching_cubes {

// edge_table and index_table are copied from public domain source code
// Marching Cubes Example Program by Cory Bloyd (corysama@yahoo.com)
//
// For a description of the algorithm go to
// http://paulbourke.net/geometry/polygonise/

// This table lists the edges intersected by the surface for all 256 possible
// vertex states. There are 12 edges to a cube. For each entry in the table, if
// edge #n is intersected, then bit #n is set to 1
static const u16 edge_table[256] =
{
	0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

// For each of the possible vertex states listed in edge_table there is a
// specific triangulation of the edge intersection points. This lists all of
// them in the form of 0-5 edge triples with the list terminated by the invalid
// value -1.
static const s8 index_table[256][16] =
{
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

static Vector3 interpolate_vertex(float isovalue, Vector3 p0, Vector3 p1, float i0, float i1)
{
	if(almost_equals(i0, i1))
	{
		return p0;
	}
	float t = unlerp(i0, i1, isovalue);
	return lerp(p0, p1, t);
}

void triangulate(Voxmap* map, Vector3 scale, u8 isovalue, Triangle** result, int* result_count)
{
	int columns = map->columns;
	int rows = map->rows;
	int slices = map->slices;

	int rc = columns * rows;
	int c = columns;

	Triangle* triangles = nullptr;
	int triangles_count = 0;
	int triangles_capacity = 0;

	for(int i = 0; i < slices - 1; ++i)
	{
		for(int j = 0; j < rows - 1; ++j)
		{
			for(int k = 0; k < columns - 1; ++k)
			{
				// Take samples of the field at each corner of the cell.
				float x[8];
				x[0] = map->voxels[rc * (i    ) + c * (j    ) + (k    )];
				x[1] = map->voxels[rc * (i + 1) + c * (j    ) + (k    )];
				x[2] = map->voxels[rc * (i + 1) + c * (j + 1) + (k    )];
				x[3] = map->voxels[rc * (i    ) + c * (j + 1) + (k    )];
				x[4] = map->voxels[rc * (i    ) + c * (j    ) + (k + 1)];
				x[5] = map->voxels[rc * (i + 1) + c * (j    ) + (k + 1)];
				x[6] = map->voxels[rc * (i + 1) + c * (j + 1) + (k + 1)];
				x[7] = map->voxels[rc * (i    ) + c * (j + 1) + (k + 1)];

				// Calculate the positions of each vertex.
				Vector3 p[8];
				p[0] = {scale.x * (k    ), scale.y * (j    ), scale.z * (i    )};
				p[1] = {scale.x * (k    ), scale.y * (j    ), scale.z * (i + 1)};
				p[2] = {scale.x * (k    ), scale.y * (j + 1), scale.z * (i + 1)};
				p[3] = {scale.x * (k    ), scale.y * (j + 1), scale.z * (i    )};
				p[4] = {scale.x * (k + 1), scale.y * (j    ), scale.z * (i    )};
				p[5] = {scale.x * (k + 1), scale.y * (j    ), scale.z * (i + 1)};
				p[6] = {scale.x * (k + 1), scale.y * (j + 1), scale.z * (i + 1)};
				p[7] = {scale.x * (k + 1), scale.y * (j + 1), scale.z * (i    )};

				// Check the value at each corner of the cell and use that to
				// build an index into the edge table.
				int cube_index = 0;
				if(x[0] < isovalue) cube_index |= 1;
				if(x[1] < isovalue) cube_index |= 2;
				if(x[2] < isovalue) cube_index |= 4;
				if(x[3] < isovalue) cube_index |= 8;
				if(x[4] < isovalue) cube_index |= 16;
				if(x[5] < isovalue) cube_index |= 32;
				if(x[6] < isovalue) cube_index |= 64;
				if(x[7] < isovalue) cube_index |= 128;

				// If any part of the surface intersects the cell, then
				// triangles can be output.
				u16 edge = edge_table[cube_index];
				if(edge != 0)
				{
					ENSURE_ARRAY_SIZE(triangles, 4);

					Vector3 vertices[12];

					// Find the vertices where the surface intersects the cell.
					if(edge & 1) vertices[0] = interpolate_vertex(isovalue, p[0], p[1], x[0], x[1]);
					if(edge & 2) vertices[1] = interpolate_vertex(isovalue, p[1], p[2], x[1], x[2]);
					if(edge & 4) vertices[2] = interpolate_vertex(isovalue, p[2], p[3], x[2], x[3]);
					if(edge & 8) vertices[3] = interpolate_vertex(isovalue, p[3], p[0], x[3], x[0]);
					if(edge & 16) vertices[4] = interpolate_vertex(isovalue, p[4], p[5], x[4], x[5]);
					if(edge & 32) vertices[5] = interpolate_vertex(isovalue, p[5], p[6], x[5], x[6]);
					if(edge & 64) vertices[6] = interpolate_vertex(isovalue, p[6], p[7], x[6], x[7]);
					if(edge & 128) vertices[7] = interpolate_vertex(isovalue, p[7], p[4], x[7], x[4]);
					if(edge & 256) vertices[8] = interpolate_vertex(isovalue, p[0], p[4], x[0], x[4]);
					if(edge & 512) vertices[9] = interpolate_vertex(isovalue, p[1], p[5], x[1], x[5]);
					if(edge & 1024) vertices[10] = interpolate_vertex(isovalue, p[2], p[6], x[2], x[6]);
					if(edge & 2048) vertices[11] = interpolate_vertex(isovalue, p[3], p[7], x[3], x[7]);

					// Output the triangles that the index table indicates for
					// this cell's configuration.
					for(int m = 0; m < 16 && index_table[cube_index][m] != -1; m += 3)
					{
						// Warning: the table's triangle indices are wound
						// counterclockwise. This is opposite the winding of the
						// desired output, so they're swapped here.
						triangles[triangles_count].vertices[0] = vertices[index_table[cube_index][m]];
						triangles[triangles_count].vertices[2] = vertices[index_table[cube_index][m + 1]];
						triangles[triangles_count].vertices[1] = vertices[index_table[cube_index][m + 2]];
						triangles_count += 1;
					}
				}
			}
		}
	}

	*result = triangles;
	*result_count = triangles_count;
}

} // namespace marching_cubes

// §3.14 Render System..........................................................

namespace render {

// §3.15.1 Shader Sources.......................................................

const char* default_vertex_source = R"(
#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 colour;

uniform mat4x4 model_view_projection;
uniform mat4x4 normal_matrix;

out vec3 surface_normal;
out vec3 surface_colour;

void main()
{
	gl_Position = model_view_projection * vec4(position, 1.0);
	surface_normal = (normal_matrix * vec4(normal, 0.0)).xyz;
	surface_colour = colour.rgb;
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
layout(location = 2) in vec4 colour;

uniform mat4x4 model_view_projection;

out vec4 surface_colour;

void main()
{
	gl_Position = model_view_projection * vec4(position, 1.0);
	surface_colour = colour;
}
)";

const char* fragment_source_vertex_colour = R"(
#version 330

layout(location = 0) out vec4 output_colour;

in vec4 surface_colour;

void main()
{
	output_colour = surface_colour;
}
)";

const char* vertex_source_texture_only = R"(
#version 330

layout(location = 0) in vec3 position;
layout(location = 3) in vec2 texcoord;

uniform mat4x4 model_view_projection;

out vec2 surface_texcoord;

void main()
{
	gl_Position = model_view_projection * vec4(position, 1.0);
	surface_texcoord = texcoord;
}
)";

const char* fragment_source_texture_only = R"(
#version 330

uniform sampler2D texture;

layout(location = 0) out vec4 output_colour;

in vec2 surface_texcoord;

void main()
{
	output_colour = vec4(texture2D(texture, surface_texcoord).rgb, 1.0);
}
)";

const char* vertex_source_camera_fade = R"(
#version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 colour;

uniform mat4x4 model_view_projection;
uniform mat4x4 normal_matrix;
uniform float near;
uniform float far;
uniform float fade_distance;

out vec3 surface_normal;
out vec3 surface_colour;
out float distance_to_camera;

void main()
{
	vec4 position = model_view_projection * vec4(position, 1.0);
	gl_Position = position;
	surface_normal = (normal_matrix * vec4(normal, 0.0)).xyz;
	surface_colour = colour.rgb;

	float z_ndc = 2.0 * (position.z / position.w) - 1.0;
	float z_eye = 2.0 * near * far / (far + near - z_ndc * (far - near));
	distance_to_camera = (1.0 / fade_distance) * (z_eye - fade_distance);
}
)";

const char* fragment_source_camera_fade = R"(
#version 330

layout(location = 0) out vec4 output_colour;

uniform sampler2D dither_pattern;
uniform float dither_pattern_side;
uniform vec3 light_direction;

in vec3 surface_normal;
in vec3 surface_colour;
in float distance_to_camera;

float half_lambert(vec3 n, vec3 l)
{
	return 0.5 * dot(n, l) + 0.5;
}

void main()
{
	if(distance_to_camera < texture2D(dither_pattern, gl_FragCoord.xy / dither_pattern_side).r)
	{
		discard;
	}
	float light = half_lambert(surface_normal, light_direction);
	output_colour = vec4(surface_colour * vec3(light), 1.0);
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
	glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, vertex_size, offset2);
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

static void object_generate_floor(Object* object, arandom::Sequence* randomness, AABB* bounds)
{
	Floor floor = {};
	AABB box = {vector3_max, vector3_min};
	for(int y = 0; y < 10; ++y)
	{
		for(int x = y & 1; x < 10; x += 2)
		{
			Vector3 bottom_left = {0.4f * (x - 10.0f), 0.4f * y, -1.4f};
			Vector3 dimensions = {0.4f, 0.4f, 0.4f};
			bool added = floor_add_box(&floor, bottom_left, dimensions, randomness);
			if(!added)
			{
				floor_destroy(&floor);
				return;
			}
			box = aabb_merge(box, {bottom_left, bottom_left + dimensions});
		}
	}
	Vector3 wall_position = {1.0f, 0.0f, -1.0f};
	Vector3 wall_dimensions = {0.1f, 2.0f, 1.0f};
	bool added = floor_add_box(&floor, wall_position, wall_dimensions, randomness);
	if(!added)
	{
		floor_destroy(&floor);
		return;
	}
	*bounds = aabb_merge(box, {wall_position, wall_position + wall_dimensions});
	object_set_surface(object, floor.vertices, floor.vertices_count, floor.indices, floor.indices_count);
	floor_destroy(&floor);
}

static void object_generate_player(Object* object, arandom::Sequence* randomness, AABB* bounds)
{
	Floor floor = {};
	Vector3 dimensions = {0.5f, 0.5f, 0.7f};
	Vector3 position = -dimensions / 2.0f;
	bounds->min = position;
	bounds->max = position + dimensions;
	bool added = floor_add_box(&floor, position, dimensions, randomness);
	if(!added)
	{
		floor_destroy(&floor);
		return;
	}
	object_set_surface(object, floor.vertices, floor.vertices_count, floor.indices, floor.indices_count);
	floor_destroy(&floor);
}

static void generate_normals_from_faces(VertexPNC* vertices, int vertices_count, u16* indices, int indices_count)
{
	int* seen = ALLOCATE(int, vertices_count);
	if(!seen)
	{
		DEALLOCATE(vertices);
		DEALLOCATE(indices);
		return;
	}
	for(int i = 0; i < indices_count; i += 3)
	{
		u16 ia = indices[i + 0];
		u16 ib = indices[i + 1];
		u16 ic = indices[i + 2];
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
}

static void object_generate_terrain(Object* object, arandom::Sequence* randomness, AABB* bounds, Triangle** triangles, int* triangles_count)
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
			float fz = arandom::float_range(randomness, -0.5f, 0.5f) - 1.0f;
			vertices[columns * y + x].position = {fx, fy, fz};
		}
	}

	AABB box = {vector3_max, vector3_min};
	for(int i = 0; i < vertices_count; ++i)
	{
		box.min = min3(box.min, vertices[i].position);
		box.max = max3(box.max, vertices[i].position);
	}
	*bounds = box;

	// Generate random vertex colours.
	for(int i = 0; i < vertices_count; ++i)
	{
		float h = arandom::float_range(randomness, 0.0f, 0.1f);
		float s = arandom::float_range(randomness, 0.7f, 0.9f);
		float l = arandom::float_range(randomness, 0.5f, 1.0f);
		Vector3 rgb = hsl_to_rgb({h, s, l});
		Vector4 rgba = make_vector4(rgb);
		vertices[i].colour = rgba_to_u32(rgba);
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

		indices[o + 0] = k + columns + 1;
		indices[o + 1] = k + columns;
		indices[o + 2] = k;

		indices[o + 3] = k + 1;
		indices[o + 4] = k + columns + 1;
		indices[o + 5] = k;
	}

	generate_normals_from_faces(vertices, vertices_count, indices, indices_count);

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
		Vector3 a = vertices[indices[3 * i + 0]].position;
		Vector3 b = vertices[indices[3 * i + 1]].position;
		Vector3 c = vertices[indices[3 * i + 2]].position;
		t[i].vertices[0] = a;
		t[i].vertices[1] = b;
		t[i].vertices[2] = c;
	}

	object_set_surface(object, vertices, vertices_count, indices, indices_count);

	DEALLOCATE(vertices);
	DEALLOCATE(indices);
}

void object_generate_sky(Object* object)
{
	const float radius = 1.0f;
	const int meridians = 9;
	const int parallels = 7;
	int rings = parallels + 1;

	int vertices_count = meridians * parallels + 2;
	VertexPNC* vertices = ALLOCATE(VertexPNC, vertices_count);
	if(!vertices)
	{
		return;
	}

	const Vector3 top_colour = {1.0f, 1.0f, 0.2f};
	const Vector3 bottom_colour = {0.1f, 0.7f, 0.6f};
	vertices[0].position = radius * vector3_unit_z;
	vertices[0].normal = -vector3_unit_z;
	vertices[0].colour = rgb_to_u32(top_colour);
	for(int i = 0; i < parallels; ++i)
	{
		float step = (i + 1) / static_cast<float>(rings);
		float theta = step * pi;
		Vector3 ring_colour = lerp(top_colour, bottom_colour, step);
		for(int j = 0; j < meridians; ++j)
		{
			float phi = (j + 1) / static_cast<float>(meridians) * tau;
			float x = radius * sin(theta) * cos(phi);
			float y = radius * sin(theta) * sin(phi);
			float z = radius * cos(theta);
			Vector3 position = {x, y, z};
			VertexPNC* vertex = &vertices[meridians * i + j + 1];
			vertex->position = position;
			vertex->normal = -normalise(position);
			vertex->colour = rgb_to_u32(ring_colour);
		}
	}
	vertices[vertices_count - 1].position = radius * -vector3_unit_z;
	vertices[vertices_count - 1].normal = vector3_unit_z;
	vertices[vertices_count - 1].colour = rgb_to_u32(bottom_colour);

	int indices_count = 6 * meridians * rings;
	u16* indices = ALLOCATE(u16, indices_count);
	if(!indices)
	{
		DEALLOCATE(vertices);
		return;
	}

	int out_base = 0;
	int in_base = 1;
	for(int i = 0; i < meridians; ++i)
	{
		int o = out_base + 3 * i;
		indices[o + 0] = 0;
		indices[o + 1] = in_base + (i + 1) % meridians;
		indices[o + 2] = in_base + i;
	}
	out_base += 3 * meridians;
	for(int i = 0; i < rings - 2; ++i)
	{
		for(int j = 0; j < meridians; ++j)
		{
			int x = meridians * i + j;
			int o = out_base + 6 * x;
			int k0 = in_base + x;
			int k1 = in_base + meridians * i;

			indices[o + 0] = k0;
			indices[o + 1] = k1 + (j + 1) % meridians;
			indices[o + 2] = k0 + meridians;

			indices[o + 3] = k0 + meridians;
			indices[o + 4] = k1 + (j + 1) % meridians;
			indices[o + 5] = k1 + meridians + (j + 1) % meridians;
		}
	}
	out_base += 6 * meridians * (rings - 2);
	in_base += meridians * (parallels - 2);
	for(int i = 0; i < meridians; ++i)
	{
		int o = out_base + 3 * i;
		indices[o + 0] = vertices_count - 1;
		indices[o + 1] = in_base + i;
		indices[o + 2] = in_base + (i + 1) % meridians;
	}

	object_set_surface(object, vertices, vertices_count, indices, indices_count);

	DEALLOCATE(vertices);
	DEALLOCATE(indices);
}

static const int point_entry_points_cap = 8;

struct PointEntry
{
	Vector3 points[point_entry_points_cap];
	int indices[point_entry_points_cap];
	int points_count;
};

struct PointTable
{
	AABB bounds;
	PointEntry** entries;
	int entries_count;
	int divisions;
};

static void point_table_create(PointTable* table, AABB bounds, int divisions)
{
	int dc = divisions * divisions * divisions;
	table->bounds = bounds;
	table->entries = ALLOCATE(PointEntry*, dc);
	table->entries_count = dc;
	table->divisions = divisions;
}

static void point_table_destroy(PointTable* table)
{
	for(int i = 0; i < table->entries_count; ++i)
	{
		DEALLOCATE(table->entries[i]);
	}
	DEALLOCATE(table->entries);
}

static int hash_point(PointTable* table, Vector3 point)
{
	AABB bounds = table->bounds;
	Vector3 p = pointwise_divide(point - bounds.min, bounds.max - bounds.min);
	int d = table->divisions;
	Vector3 v = (d - 1) * p;
	int ix = round(v.x);
	int iy = round(v.y);
	int iz = round(v.z);
	int index = (d * d * iz) + (d * iy) + ix;
	return index % table->entries_count;
}

static bool vector3_close(Vector3 v0, Vector3 v1)
{
	return
		almost_equals(v0.x, v1.x) &&
		almost_equals(v0.y, v1.y) &&
		almost_equals(v0.z, v1.z);
}

static void insert(PointTable* table, Vector3 point, int index)
{
	int entry_index = hash_point(table, point);
	PointEntry* entry = table->entries[entry_index];
	if(!entry)
	{
		entry = ALLOCATE(PointEntry, 1);
		table->entries[entry_index] = entry;
	}

	entry->points[entry->points_count] = point;
	entry->indices[entry->points_count] = index;
	entry->points_count += 1;
	ASSERT(entry->points_count < point_entry_points_cap);
}

static bool find_index(PointTable* table, Vector3 point, int* index)
{
	int entry_index = hash_point(table, point);
	PointEntry* entry = table->entries[entry_index];
	if(!entry)
	{
		return false;
	}

	for(int i = 0; i < entry->points_count; ++i)
	{
		if(vector3_close(point, entry->points[i]))
		{
			*index = entry->indices[i];
			return true;
		}
	}

	return false;
}

static void object_copy_triangles(Object* object, AABB* bounds, Triangle* triangles, int triangles_count)
{
	const u32 colour = rgba_to_u32(colour_cyan);

	// The bounds will be useful for indexing the vertices, so compute that
	// first, even though it involves redundant vertex checks.
	AABB box = {vector3_max, vector3_min};
	for(int i = 0; i < triangles_count; ++i)
	{
		AABB triangle_bounds = aabb_from_triangle(&triangles[i]);
		box = aabb_merge(box, triangle_bounds);
	}

	// Allocate the maximum amount of space even though the finished vertices
	// will be fewer.
	int vertices_capacity = 3 * triangles_count;
	VertexPNC* vertices = ALLOCATE(VertexPNC, vertices_capacity);

	// The indices for sure won't change in number.
	int indices_count = vertices_capacity;
	u16* indices = ALLOCATE(u16, indices_count);

	// Assign indices to unique vertices and output positions and colour.
	PointTable table;
	point_table_create(&table, box, 24);
	u16 index = 0;
	for(int i = 0; i < triangles_count; ++i)
	{
		for(int j = 0; j < 3; ++j)
		{
			Vector3 p = triangles[i].vertices[j];
			int found_index;
			if(find_index(&table, p, &found_index))
			{
				indices[3 * i + j] = found_index;
			}
			else
			{
				insert(&table, p, index);
				vertices[index].position = p;
				vertices[index].colour = colour;
				indices[3 * i + j] = index;
				index += 1;
			}
		}
	}
	point_table_destroy(&table);
	int vertices_count = index;

	generate_normals_from_faces(vertices, vertices_count, indices, indices_count);

	object_set_surface(object, vertices, vertices_count, indices, indices_count);

	DEALLOCATE(vertices);
	DEALLOCATE(indices);

	*bounds = box;
}

// Whole system

struct CallList
{
	int indices[8];
	int count;
};

struct
{
	GLuint program;
	GLint light_direction;
	GLint model_view_projection;
	GLint normal_matrix;
} shader_default;

struct
{
	GLuint program;
	GLint model_view_projection;
	GLint normal_matrix;
	GLint light_direction;
	GLint dither_pattern;
	GLint dither_pattern_side;
	GLint near;
	GLint far;
	GLint fade_distance;
} shader_camera_fade;

struct
{
	GLuint program;
	GLint model_view_projection;
} shader_vertex_colour;

struct
{
	GLuint program;
	GLint model_view_projection;
	GLint texture;
} shader_texture_only;

GLuint camera_fade_dither_pattern;
GLuint nearest_repeat;
Matrix4 projection;
Matrix4 sky_projection;
Matrix4 screen_projection;
arandom::Sequence randomness;
int objects_count = 5;
Object objects[5];
AABB objects_bounds[5];
Object sky;
Triangle* terrain_triangles;
int terrain_triangles_count;
CallList solid_calls;
CallList fade_calls;
Oscilloscope oscilloscope;
const int traces_count = 2;
Trace traces[traces_count];
Vector2* particles;
int particles_count;
LineSegment* isolines[5];
int isolines_count[5];

TWEAKER_BOOL(debug_draw_colliders, false);
TWEAKER_INT_RANGE(debug_bih_tree_depth, 0, -1, 16);
TWEAKER_BOOL(debug_show_oscilloscope, false);
TWEAKER_BOOL(debug_show_texture_gallery, false);

const float near_plane = 0.05f;
const float far_plane = 12.0f;

static bool system_initialise()
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	arandom::seed_by_time(&randomness);

	Object tetrahedron;
	Vector3 positions[4] =
	{
		{+1.0f, +0.0f, -1.0f / sqrt(2.0f)},
		{-1.0f, +0.0f, -1.0f / sqrt(2.0f)},
		{+0.0f, +1.0f, +1.0f / sqrt(2.0f)},
		{+0.0f, -1.0f, +1.0f / sqrt(2.0f)},
	};
	VertexPNC vertices[4] =
	{
		{positions[0], { 0.816497f, 0.0f, -0.57735f}, 0xffffffff},
		{positions[1], {-0.816497f, 0.0f, -0.57735f}, 0xffffffff},
		{positions[2], { 0.0f, 0.816497f, +0.57735f}, 0xffffffff},
		{positions[3], { 0.0f,-0.816497f, +0.57735f}, 0xffffffff},
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
	objects_bounds[0] = compute_bounds(positions, 4);

	Object floor;
	object_create(&floor);
	object_generate_floor(&floor, &randomness, &objects_bounds[1]);
	objects[1] = floor;

	Object player;
	object_create(&player);
	object_generate_player(&player, &randomness, &objects_bounds[2]);
	objects[2] = player;

	Object terrain;
	object_create(&terrain);
	object_generate_terrain(&terrain, &randomness, &objects_bounds[3], &terrain_triangles, &terrain_triangles_count);
	objects[3] = terrain;

	Object metaballs;
	object_create(&metaballs);
	{
		const int side = 24;
		Voxmap canvas;
		canvas.columns = side;
		canvas.rows = side;
		canvas.slices = side;
		canvas.voxels = ALLOCATE(u8, side * side * side);

		const int brush_side = 4;
		Voxmap brush;
		brush.columns = brush_side;
		brush.rows = brush_side;
		brush.slices = brush_side;
		brush.voxels = ALLOCATE(u8, brush_side * brush_side * brush_side);
		draw_checkers(&brush);

		Vector3* spline;
		int spline_count;
		generate_random_spline(&randomness, &spline, &spline_count);

		for(int i = 0; i < spline_count; ++i)
		{
			Vector3 tip = spline[i];

			Vector3 axis;
			axis.x = arandom::float_range(&randomness, -1.0f, 1.0f);
			axis.y = arandom::float_range(&randomness, -1.0f, 1.0f);
			axis.z = arandom::float_range(&randomness, -1.0f, 1.0f);
			float angle = arandom::float_range(&randomness, 0.0f, tau);
			Quaternion rotation = axis_angle_rotation(axis, angle);

			Vector3 scale = {1.2f, 1.2f, 1.2f};

			Matrix4 transform = compose_transform(tip, rotation, scale);

			draw_voxmap(&canvas, &brush, transform);
		}

		DEALLOCATE(spline);

		DEALLOCATE(brush.voxels);

		Triangle* triangles;
		int triangles_count;
		Vector3 scale = {0.1f, 0.1f, 0.1f};
		u8 isovalue = 128;
		marching_cubes::triangulate(&canvas, scale, isovalue, &triangles, &triangles_count);

		DEALLOCATE(canvas.voxels);

		object_copy_triangles(&metaballs, &objects_bounds[4], triangles, triangles_count);

		DEALLOCATE(triangles);
	}
	objects[4] = metaballs;

	object_create(&sky);
	object_generate_sky(&sky);

	// Setup samplers.
	{
		glGenSamplers(1, &nearest_repeat);
		glSamplerParameteri(nearest_repeat, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glSamplerParameteri(nearest_repeat, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glSamplerParameteri(nearest_repeat, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glSamplerParameteri(nearest_repeat, GL_TEXTURE_WRAP_T, GL_REPEAT);
	}

	shader_default.program = load_shader_program(default_vertex_source, default_fragment_source);
	if(shader_default.program == 0)
	{
		LOG_ERROR("The default shader failed to load.");
		return false;
	}
	{
		GLuint program = shader_default.program;
		shader_default.model_view_projection = glGetUniformLocation(program, "model_view_projection");
		shader_default.normal_matrix = glGetUniformLocation(program, "normal_matrix");
		shader_default.light_direction = glGetUniformLocation(program, "light_direction");
	}

	shader_vertex_colour.program = load_shader_program(vertex_source_vertex_colour, fragment_source_vertex_colour);
	if(shader_vertex_colour.program == 0)
	{
		LOG_ERROR("The vertex colour shader failed to load.");
		return false;
	}
	{
		shader_vertex_colour.model_view_projection = glGetUniformLocation(shader_vertex_colour.program, "model_view_projection");
	}

	shader_texture_only.program = load_shader_program(vertex_source_texture_only, fragment_source_texture_only);
	if(shader_texture_only.program == 0)
	{
		LOG_ERROR("The texture-only shader failed to load.");
		return false;
	}
	{
		GLuint program = shader_texture_only.program;
		shader_texture_only.model_view_projection = glGetUniformLocation(program, "model_view_projection");
		shader_texture_only.texture = glGetUniformLocation(program, "texture");

		glUseProgram(shader_texture_only.program);
		glUniform1i(shader_texture_only.texture, 0);
	}

	shader_camera_fade.program = load_shader_program(vertex_source_camera_fade, fragment_source_camera_fade);
	if(shader_camera_fade.program == 0)
	{
		LOG_ERROR("The camera fade shader failed to load.");
		return false;
	}
	{
		GLuint program = shader_camera_fade.program;
		shader_camera_fade.model_view_projection = glGetUniformLocation(program, "model_view_projection");
		shader_camera_fade.normal_matrix = glGetUniformLocation(program, "normal_matrix");
		shader_camera_fade.light_direction = glGetUniformLocation(program, "light_direction");
		shader_camera_fade.dither_pattern = glGetUniformLocation(program, "dither_pattern");
		shader_camera_fade.dither_pattern_side = glGetUniformLocation(program, "dither_pattern_side");
		shader_camera_fade.near = glGetUniformLocation(program, "near");
		shader_camera_fade.far = glGetUniformLocation(program, "far");
		shader_camera_fade.fade_distance = glGetUniformLocation(program, "fade_distance");

		glUseProgram(shader_camera_fade.program);
		glUniform1i(shader_camera_fade.dither_pattern, 0);
		glUniform1f(shader_camera_fade.dither_pattern_side, 4.0f);
		glUniform1f(shader_camera_fade.near, near_plane);
		glUniform1f(shader_camera_fade.far, far_plane);
		glUniform1f(shader_camera_fade.fade_distance, 0.2f);

		float pattern[16] =
		{
			 0.0000f, 0.0005f, 0.0125f, 0.0625f,
			 0.0075f, 0.0025f, 0.0875f, 0.0375f,
			 0.1875f, 0.6875f, 0.0625f, 0.5625f,
			 0.9375f, 0.4375f, 0.8125f, 0.3125f,
		};
		glGenTextures(1, &camera_fade_dither_pattern);
		glBindTexture(GL_TEXTURE_2D, camera_fade_dither_pattern);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 4, 4, 0, GL_RED, GL_FLOAT, pattern);

		glActiveTexture(GL_TEXTURE0 + 0);
		glBindTexture(GL_TEXTURE_2D, camera_fade_dither_pattern);
		glBindSampler(0, nearest_repeat);
	}

	immediate::context_create();
	immediate::context->shaders[0] = shader_vertex_colour.program;
	immediate::context->shaders[1] = shader_texture_only.program;

	oscilloscope_default(&oscilloscope);

	{
		const float side = 5.0f;
		int noise_square_count = 512;
		Vector2* noise_square = create_blue_noise(noise_square_count, side, &randomness);
		particles = ALLOCATE(Vector2, noise_square_count);
#if 0
		const int map_side = 20;
		SpashCell* map = ALLOCATE(SpashCell, map_side * map_side);
		for(int i = 0; i < particles_count; ++i)
		{
			int x = (map_side - 1) * (particles[i].x / side);
			int y = (map_side - 1) * (particles[i].y / side);
			SpashCell* cell = &map[map_side * y + x];
			cell->indices[cell->count] = i;
			cell->count = MIN(cell->count + 1, spash_cell_indices_capacity - 1);
			ASSERT(cell->count < spash_cell_indices_capacity - 1);
		}
		SAFE_DEALLOCATE(map);
#endif
		particles_count = 0;
#if 1
		int map_side = 64;
		u8* map = ALLOCATE(u8, map_side * map_side);
		for(int y = 0; y < map_side; ++y)
		{
			for(int x = 0; x < map_side; ++x)
			{
				float s0 = cos(3.0f * tau * x / static_cast<float>(map_side));
				float s1 = sin(3.0f * tau * y / static_cast<float>(map_side));
				map[map_side * y + x] = (s0 + s1 > 0.0f) ? 0 : 255;
			}
		}

		for(int i = 0; i < noise_square_count; ++i)
		{
			Vector2 p = noise_square[i];
			int x = map_side * p.x / side;
			int y = map_side * p.y / side;
			float m = map[map_side * y + x] / 255.0f;
			if(arandom::float_range(&randomness, 0.0f, 1.0f) < m)
			{
				particles[particles_count] = p;
				particles_count += 1;
			}
		}
#else
		Vector2 center = {side / 2.0f, side / 2.0f};
		const float ri = 1.0f;
		const float ro = 2.5f;
		for(int i = 0; i < noise_square_count; ++i)
		{
			Vector2 p = noise_square[i];
			Vector2 v = p - center;
			float d = squared_length(v);
			float ris = ri * ri;
			float ros = ro * ro;
			float m = (d - ris) / (ros - ris);
			if(d < ris || (d < ros && arandom::float_range(0.0f, 1.0f) > m))
			{
				particles[particles_count] = p;
				particles_count += 1;
			}
		}
#endif

		SAFE_DEALLOCATE(noise_square);
	}

	// Make 2D metaballs.
	{
		const int side = 24;
		Floatmap map;
		map.columns = side;
		map.rows = side;
		map.values = ALLOCATE(float, side * side);
		marching_squares::draw_metaballs(&map, &randomness);

		for(int i = 0; i < 5; ++i)
		{
			float isovalue = 0.14f * i + 0.1f;
			Vector2 scale = {0.1f, 0.1f};
			marching_squares::delineate(&map, isovalue, scale, &isolines[i], &isolines_count[i]);
		}

		DEALLOCATE(map.values);
	}

	return true;
}

static void system_terminate(bool functions_loaded)
{
	if(functions_loaded)
	{
		for(int i = 0; i < objects_count; ++i)
		{
			object_destroy(&objects[i]);
		}
		glDeleteSamplers(1, &nearest_repeat);
		glDeleteProgram(shader_default.program);
		glDeleteProgram(shader_vertex_colour.program);
		glDeleteProgram(shader_texture_only.program);
		glDeleteProgram(shader_camera_fade.program);
		glDeleteTextures(1, &camera_fade_dither_pattern);
		immediate::context_destroy();
		SAFE_DEALLOCATE(particles);
		for(int i = 0; i < 5; ++i)
		{
			SAFE_DEALLOCATE(isolines[i]);
		}
	}
}

static void resize_viewport(int width, int height)
{
	const float fov = pi_over_2 * (2.0f / 3.0f);
	projection = perspective_projection_matrix(fov, width, height, near_plane, far_plane);
	sky_projection = perspective_projection_matrix(fov, width, height, 0.001f, 1.0f);
	screen_projection = orthographic_projection_matrix(width, height, -1.0f, 1.0f);
	glViewport(0, 0, width, height);
}

static void system_update(Vector3 position, Vector3 dancer_position, World* world, Tweaker* tweaker, profile::Inspector* inspector)
{
	PROFILE_SCOPED();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	const Vector3 camera_position = {0.0f, -3.5f, 1.5f};

	Matrix4 models[6];

	// Set up the matrices.
	Matrix4 view_projection;
	{
		const Vector3 scale = vector3_one;

		Matrix4 model0;
		{
			static float angle = 0.0f;
			angle += 0.02f;

			Vector3 where = {0.0f, 2.0f, 0.0f};
			where += dancer_position;
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
		Matrix4 model4 = matrix4_identity;
		Matrix4 model5 = matrix4_identity;

		models[0] = model0;
		models[1] = model1;
		models[2] = model2;
		models[3] = model3;
		models[4] = model4;
		models[5] = model5;

		const Vector3 camera_target = {0.0f, 0.0f, 0.5f};
		const Matrix4 view = look_at_matrix(camera_position, camera_target, vector3_unit_z);

		Vector3 direction = normalise(camera_target - camera_position);
		const Matrix4 view5 = look_at_matrix(vector3_zero, direction, vector3_unit_z);

		object_set_matrices(&objects[0], model0, view, projection);
		object_set_matrices(&objects[1], model1, view, projection);
		object_set_matrices(&objects[2], model2, view, projection);
		object_set_matrices(&objects[3], model3, view, projection);
		object_set_matrices(&objects[4], model4, view, projection);
		object_set_matrices(&sky, model5, view5, sky_projection);

		immediate::set_matrices(view, projection);

		Vector3 light_direction = {0.7f, 0.4f, -1.0f};
		light_direction = normalise(-(view * light_direction));

		glUseProgram(shader_default.program);
		glUniform3fv(shader_default.light_direction, 1, &light_direction[0]);

		glUseProgram(shader_camera_fade.program);
		glUniform3fv(shader_camera_fade.light_direction, 1, &light_direction[0]);

		view_projection = projection * view;
	}

	// Form lists of objects not culled.
	PROFILE_BEGIN_NAMED("cull_and_list");

	fade_calls.count = 0;
	solid_calls.count = 0;
	for(int i = 0; i < objects_count; ++i)
	{
		Frustum frustum = make_frustum(objects[i].model_view_projection);
		if(intersect_aabb_frustum(&frustum, &objects_bounds[i]))
		{
			Vector3 camera = inverse_transform(models[i]) * camera_position;
			if(distance_point_to_aabb(objects_bounds[i], camera) < 1.0f)
			{
				fade_calls.indices[fade_calls.count] = i;
				fade_calls.count += 1;
			}
			else
			{
				solid_calls.indices[solid_calls.count] = i;
				solid_calls.count += 1;
			}
		}
	}

	PROFILE_END();

	// Draw all the faded objects first because they are close to the camera.
	PROFILE_BEGIN_NAMED("faded_phase");

	glUseProgram(shader_camera_fade.program);
	for(int i = 0; i < fade_calls.count; ++i)
	{
		Object* o = &objects[fade_calls.indices[i]];
		glUniformMatrix4fv(shader_camera_fade.model_view_projection, 1, GL_TRUE, o->model_view_projection.elements);
		glUniformMatrix4fv(shader_camera_fade.normal_matrix, 1, GL_TRUE, o->normal_matrix.elements);
		glBindVertexArray(o->vertex_array);
		glDrawElements(GL_TRIANGLES, o->indices_count, GL_UNSIGNED_SHORT, nullptr);
	}

	PROFILE_END();

	// Draw all the solid objects in the list.
	PROFILE_BEGIN_NAMED("solid_phase");

	glUseProgram(shader_default.program);
	for(int i = 0; i < solid_calls.count; ++i)
	{
		Object* o = &objects[solid_calls.indices[i]];
		glUniformMatrix4fv(shader_default.model_view_projection, 1, GL_TRUE, o->model_view_projection.elements);
		glUniformMatrix4fv(shader_default.normal_matrix, 1, GL_TRUE, o->normal_matrix.elements);
		glBindVertexArray(o->vertex_array);
		glDrawElements(GL_TRIANGLES, o->indices_count, GL_UNSIGNED_SHORT, nullptr);
	}

	PROFILE_END();

	// Draw debug visualisers.
	immediate::draw();

	if(debug_draw_colliders)
	{
		immediate::add_wire_ellipsoid(position, {0.3f, 0.3f, 0.5f}, {1.0f, 0.0f, 1.0f});
		immediate::draw();

		for(int i = 0; i < world->colliders_count; ++i)
		{
			draw_bih_tree(&world->colliders[i].tree, debug_bih_tree_depth);
		}
	}

	// Blue noise test
	{
		Vector3 bottom_left = {-3.0f, -1.0f, 0.0f};
		draw_particles(particles, particles_count, bottom_left);
	}

	// Isolines test
	{
		for(int i = 0; i < 5; ++i)
		{
			for(int j = 0; j < isolines_count[i]; ++j)
			{
				LineSegment line = isolines[i][j];
				Vector3 start = {line.vertices[0].x - 2.0f, 0.0f, line.vertices[0].y};
				Vector3 end = {line.vertices[1].x - 2.0f, 0.0f, line.vertices[1].y};
				immediate::add_line(start, end, colour_red);
			}
			immediate::draw();
		}
	}

	// Draw the sky behind everything else.
	{
		glDepthMask(GL_FALSE);
		glUseProgram(shader_vertex_colour.program);
		Object* o = &sky;
		GLint location = shader_vertex_colour.model_view_projection;
		glUniformMatrix4fv(location, 1, GL_TRUE, o->model_view_projection.elements);
		glBindVertexArray(o->vertex_array);
		glDrawElements(GL_TRIANGLES, o->indices_count, GL_UNSIGNED_SHORT, nullptr);
		glDepthMask(GL_TRUE);
	}

	// Draw screen-space debug UI.
	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);

	Matrix4 screen_view = matrix4_identity;
	immediate::set_matrices(screen_view, screen_projection);

	if(debug_show_oscilloscope)
	{
		float y[2] = {-128.0f, 128.0f};
		for(int i = 0; i < traces_count; ++i)
		{
			trace_oscilloscope_channel(&traces[i], &oscilloscope, i);
			draw_trace(&traces[i], -256.0, y[i], 512.0f, 128.0f);
		}
	}
	else
	{
		// Maintain the traces' continuity even though they're not drawn.
		for(int i = 0; i < traces_count; ++i)
		{
			trace_oscilloscope_channel(&traces[i], &oscilloscope, i);
		}
	}

	if(debug_show_texture_gallery)
	{
		const float side = 64.0f;
		Rect rect;
		rect.bottom_left = {-350.0f, -250.0f};
		rect.dimensions = {side, side};
		Quad quad = rect_to_quad(rect);
		immediate::add_quad_textured(&quad);
		immediate::draw();
	}

	draw_tweaker(tweaker);
	profile::inspector_draw(inspector);

	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
}

} // namespace render

// 4. Audio.....................................................................

namespace audio {

enum Format
{
	FORMAT_U8,  // Unsigned 8-bit Integer
	FORMAT_S8,  // Signed 8-bit Integer
	FORMAT_U16, // Unsigned 16-bit Integer
	FORMAT_S16, // Signed 16-bit Integer
	FORMAT_U24, // Unsigned 24-bit Integer
	FORMAT_S24, // Signed 24-bit Integer
	FORMAT_U32, // Unsigned 32-bit Integer
	FORMAT_S32, // Signed 32-bit Integer
	FORMAT_F32, // 32-bit Floating-point
	FORMAT_F64, // 64-bit Floating-point
};

static int format_byte_count(Format format)
{
	switch(format)
	{
		case FORMAT_U8:
		case FORMAT_S8:
			return 1;

		case FORMAT_U16:
		case FORMAT_S16:
			return 2;

		case FORMAT_U24:
		case FORMAT_S24:
			return 3;

		case FORMAT_U32:
		case FORMAT_S32:
		case FORMAT_F32:
			return 4;

		case FORMAT_F64:
			return 8;
	}
	return 0;
}

struct DeviceDescription
{
	u64 size;
	u64 frames;
	Format format;
	u32 sample_rate;
	u8 channels;
	u8 silence;
};

static u8 get_silence_by_format(Format format)
{
	switch(format)
	{
		case FORMAT_U8: return 0x80;
		default:        return 0x00;
	}
}

static void fill_remaining_device_description(DeviceDescription* description)
{
	description->silence = get_silence_by_format(description->format);
	description->size = format_byte_count(description->format) * description->channels * description->frames;
}

static void fill_with_silence(float* samples, u8 silence, u64 count)
{
	memset(samples, silence, sizeof(float) * count);
}

static float pitch_to_frequency(int pitch)
{
	return 440.0f * pow(2.0f, static_cast<float>(pitch - 69) / 12.0f);
}

static float frequency_given_interval(float frequency, int semitones)
{
	return frequency * pow(2.0f, semitones / 12.0f);
}

// §4.1 Format Conversion.......................................................

inline s8 convert_to_s8(float value)
{
	return value * 127.5f - 0.5f;
}

inline s16 convert_to_s16(float value)
{
	return value * 32767.5f - 0.5f;
}

inline s32 convert_to_s32(float value)
{
	return value * 2147483647.5f - 0.5f;
}

inline float convert_to_float(float value)
{
	return value;
}

inline double convert_to_double(float value)
{
	return value;
}

struct ConversionInfo
{
	struct
	{
		Format format;
		int stride;
	} in, out;
	int channels;
};

#define DEFINE_CONVERT_BUFFER(type)\
	static void convert_buffer_to_##type(const float* in, type* out, int frames, ConversionInfo* info)\
	{\
		for(int i = 0; i < frames; ++i)\
		{\
			for(int j = 0; j < info->channels; ++j)\
			{\
				out[j] = convert_to_##type(in[j]);\
			}\
			in += info->in.stride;\
			out += info->out.stride;\
		}\
	}

DEFINE_CONVERT_BUFFER(s8);
DEFINE_CONVERT_BUFFER(s16);
DEFINE_CONVERT_BUFFER(s32);
DEFINE_CONVERT_BUFFER(float);
DEFINE_CONVERT_BUFFER(double);

static void format_buffer_from_float(float* in_samples, void* out_samples, int frames, ConversionInfo* info)
{
	switch(info->in.format)
	{
		case FORMAT_S8:
			convert_buffer_to_s8(in_samples, static_cast<s8*>(out_samples), frames, info);
			break;
		case FORMAT_S16:
			convert_buffer_to_s16(in_samples, static_cast<s16*>(out_samples), frames, info);
			break;
		case FORMAT_S32:
			convert_buffer_to_s32(in_samples, static_cast<s32*>(out_samples), frames, info);
			break;
		case FORMAT_F32:
			convert_buffer_to_float(in_samples, static_cast<float*>(out_samples), frames, info);
			break;
		case FORMAT_F64:
			convert_buffer_to_double(in_samples, static_cast<double*>(out_samples), frames, info);
			break;
		default:
			ASSERT(false);
			break;
	}
}

// §4.2 Oscillators.............................................................

// ┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐
// └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─┘ └
float square_wave(float x)
{
	return 4.0f * floor(x) - 2.0f * floor(2.0f * x) + 1.0f;
}

// ┐  ┌┐  ┌┐  ┌┐  ┌┐  ┌┐  ┌┐  ┌┐
// └──┘└──┘└──┘└──┘└──┘└──┘└──┘└
float pulse_wave(float x, float t)
{
	return 2.0f * static_cast<float>(signbit(x - floor(x) - t)) - 1.0f;
}

// ╱│ ╱│ ╱│ ╱│ ╱│ ╱│ ╱│ ╱│ ╱│ ╱│
//  │╱ │╱ │╱ │╱ │╱ │╱ │╱ │╱ │╱
float sawtooth_wave(float x)
{
	return 2.0f * (x - floor(0.5f + x));
}

// ╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲
//  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱
float triangle_wave(float x)
{
	return abs(4.0f * fmod(x, 1.0f) - 2.0f) - 1.0f;
}

float rectified_sin(float x)
{
	return 2.0f * abs(sin(x / 2.0f)) - 1.0f;
}

float cycloid(float x)
{
	// Use slightly curtate cycloid parameters, so there's no non-differentiable
	// cusps.
	const float a = 1.0f;
	const float b = 0.95f;
	// Approximates parameter t given the value x by applying the Newton-Raphson
	// method to the equation f(t) = at - bsin(t) - x.
	float t = x;
	for(int i = 0; i < 5; ++i)
	{
		float ft = (a * t) - (b * sin(t)) - x;
		float dft = a - (b * cos(t));
		t -= ft / dft;
	}
	// Now that t is known, insert it into the parametric equation to get y,
	// then remap y to the range [-1,1].
	float y = a - (b * cos(t));
	return y / a - 1.0f;
}

// §4.3 Envelopes...............................................................

// Attack-Decay-Sustain-Release envelope
struct ADSR
{
	enum class State
	{
		Neutral,
		Attack,
		Decay,
		Sustain,
		Release,
		Post_Release,
	};

	State state;
	float prior;
	float sustain;
	float attack_base, attack_coef;
	float decay_base, decay_coef;
	float release_base, release_coef;
	float attack_rate;
	float decay_rate;
	float release_rate;
	float ratio_attack;
	float ratio_decay_release;
};

static float envelope_curve(float ratio, float rate)
{
	if(rate <= 0.0f)
	{
		return 0.0f;
	}
	else
	{
		return exp(-log((1.0f + ratio) / ratio) / rate);
	}
}

static void envelope_set_attack(ADSR* envelope, float rate)
{
	envelope->attack_rate = rate;
	float ratio = envelope->ratio_attack;
	float coef = envelope_curve(ratio, rate);
	envelope->attack_coef = coef;
	envelope->attack_base = (1.0f + ratio) * (1.0f - coef);
}

static void envelope_set_decay(ADSR* envelope, float rate)
{
	envelope->decay_rate = rate;
	float ratio = envelope->ratio_decay_release;
	float coef = envelope_curve(ratio, rate);
	envelope->decay_coef = coef;
	envelope->decay_base = (envelope->sustain - ratio) * (1.0f - coef);
}

static void envelope_set_sustain(ADSR* envelope, float sustain)
{
	envelope->sustain = sustain;
	// Recalculate the decay base, since it's the only value affected by
	// sustain.
	envelope->decay_base = (sustain - envelope->ratio_decay_release) * (1.0f - envelope->decay_coef);
}

static void envelope_set_release(ADSR* envelope, float rate)
{
	envelope->release_rate = rate;
	float ratio = envelope->ratio_decay_release;
	float coef = envelope_curve(ratio, rate);
	envelope->release_coef = coef;
	envelope->release_base = -ratio * (1.0f - coef);
}

static void envelope_set_ratio_attack(ADSR* envelope, float ratio)
{
	envelope->ratio_attack = ratio;
	// Recalculate the attack curve with the new target ratio.
	float coef = envelope_curve(ratio, envelope->attack_rate);
	envelope->attack_coef = coef;
	envelope->attack_base = (1.0f + ratio) * (1.0f - coef);
}

static void envelope_set_ratio_decay_release(ADSR* envelope, float ratio)
{
	envelope->ratio_decay_release = ratio;
	// Recalculate the decay and release curves with the new target ratio.
	float coef = envelope_curve(ratio, envelope->decay_rate);
	envelope->decay_coef = coef;
	envelope->decay_base = (envelope->sustain - ratio) * (1.0f - coef);
	coef = envelope_curve(ratio, envelope->release_rate);
	envelope->release_coef = coef;
	envelope->release_base = -ratio * (1.0f - coef);
}

static void envelope_reset(ADSR* envelope)
{
	envelope->state = ADSR::State::Neutral;
	envelope->prior = 0.0f;
}

static void envelope_setup(ADSR* envelope)
{
	envelope->ratio_attack = 0.3f;
	envelope->ratio_decay_release = 0.0001f;
	envelope_set_attack(envelope, 0.0f);
	envelope_set_decay(envelope, 0.0f);
	envelope_set_sustain(envelope, 1.0f);
	envelope_set_release(envelope, 0.0f);
	envelope_reset(envelope);
}

static float envelope_apply(ADSR* envelope)
{
	float result;
	switch(envelope->state)
	{
		case ADSR::State::Attack:
		{
			result = envelope->attack_base + envelope->prior * envelope->attack_coef;
			if(result >= 1.0f)
			{
				result = 1.0f;
				envelope->state = ADSR::State::Decay;
			}
			break;
		}
		case ADSR::State::Decay:
		{
			result = envelope->decay_base + envelope->prior * envelope->decay_coef;
			if(result <= envelope->sustain)
			{
				result = envelope->sustain;
				envelope->state = ADSR::State::Sustain;
			}
			break;
		}
		case ADSR::State::Release:
		{
			result = envelope->release_base + envelope->prior * envelope->release_coef;
			if(result <= 0.05f)
			{
				result = -0.00001f + 0.995 * envelope->prior;
				envelope->state = ADSR::State::Post_Release;
			}
			break;
		}
		case ADSR::State::Post_Release:
		{
			// This state is to reduce clicks by enforcing a ramp down to zero
			// that's soft even when the release ramp is fairly steep.
			result = -0.00001f + 0.995 * envelope->prior;
			if(result <= 0.0f)
			{
				envelope->prior = 0.0f;
				envelope->state = ADSR::State::Neutral;
			}
			break;
		}
		case ADSR::State::Sustain:
		case ADSR::State::Neutral:
		default:
		{
			result = envelope->prior;
			break;
		}
	}
	envelope->prior = result;
	return result;
}

static void envelope_gate(ADSR* envelope, bool gate)
{
	if(gate)
	{
		envelope->state = ADSR::State::Attack;
	}
	else if(envelope->state != ADSR::State::Neutral)
	{
		envelope->state = ADSR::State::Release;
	}
}

// §4.4 Filters.................................................................

// Low-Pass Filter
struct LPF
{
	float beta;
	float prior;
};

static void lpf_reset(LPF* filter)
{
	filter->prior = 0.0f;
}

static void lpf_set_corner_frequency(LPF* filter, float corner_frequency, float delta_time)
{
	float time_constant = 1.0f / (tau * corner_frequency);
	filter->beta = delta_time / (delta_time + time_constant);
}

static float lpf_apply(LPF* filter, float sample)
{
	float result = filter->prior - filter->beta * (filter->prior - sample);
	filter->prior = result;
	return result;
}

// High-Pass Filter
struct HPF
{
	float alpha;
	float prior_filtered;
	float prior_raw;
};

static void hpf_reset(HPF* filter)
{
	filter->prior_filtered = 0.0f;
	filter->prior_raw = 0.0f;
}

static void hpf_set_corner_frequency(HPF* filter, float corner_frequency, float delta_time)
{
	float time_constant = 1.0f / (tau * corner_frequency);
	filter->alpha = time_constant / (time_constant + delta_time);
}

static float hpf_apply(HPF* filter, float sample)
{
	float result = filter->alpha * (filter->prior_filtered + sample - filter->prior_raw);
	filter->prior_raw = sample;
	filter->prior_filtered = result;
	return result;
}

// Band-Pass Filter
struct BPF
{
	struct
	{
		float alpha;
		float prior;
	} low, high;
};

static void bpf_reset(BPF* filter)
{
	filter->low.prior = 0.0f;
	filter->high.prior = 0.0f;
}

static void bpf_set_passband(BPF* filter, float corner_frequency_low, float corner_frequency_high, float delta_time)
{
	ASSERT(corner_frequency_low > 0.0f);
	float time_constant = 1.0f / (tau * corner_frequency_low);
	filter->low.alpha = delta_time / (delta_time + time_constant);

	ASSERT(corner_frequency_high > 0.0f);
	time_constant = 1.0f / (tau * corner_frequency_high);
	filter->high.alpha = delta_time / (delta_time + time_constant);
}

static float bpf_apply(BPF* filter, float sample)
{
	filter->low.prior = lerp(filter->low.prior, sample, filter->low.alpha);
	filter->high.prior = lerp(filter->high.prior, sample, filter->high.alpha);
	return filter->high.prior - filter->low.prior;
}

// Tapped Delay Line
struct TDL
{
	float* buffer;
	int buffer_capacity;
	int index;
};

static void tdl_create(TDL* line, int capacity)
{
	line->buffer_capacity = capacity;
	line->buffer = ALLOCATE(float, line->buffer_capacity);
	line->index = capacity - 1;
}

static void tdl_destroy(TDL* line)
{
	SAFE_DEALLOCATE(line->buffer);
}

static void tdl_record(TDL* line, float sample)
{
	line->index = (line->index + line->buffer_capacity - 1) % line->buffer_capacity;
	line->buffer[line->index] = sample;
}

static float tdl_tap(TDL* line, float delay_time)
{
	int delay_int = delay_time;
	float delay_fraction = delay_time - static_cast<float>(delay_int);

	int next = (line->index + delay_int) % line->buffer_capacity;
	float a = line->buffer[next];
	float b = line->buffer[(next + 1) % line->buffer_capacity];

	return lerp(a, b, delay_fraction);
}

static float tdl_tap_last(TDL* line)
{
	int index = (line->index + line->buffer_capacity - 1) % line->buffer_capacity;
	return line->buffer[index];
}

// All-Pass Filter
struct APF
{
	TDL delay_line;
	float gain;
};

void apf_create(APF* filter, int max_delay)
{
	tdl_create(&filter->delay_line, max_delay);
	filter->gain = 0.2f;
}

void apf_destroy(APF* filter)
{
	tdl_destroy(&filter->delay_line);
}

static float apf_apply(APF* filter, float sample)
{
	float prior = tdl_tap_last(&filter->delay_line);
	float feedback = sample - (filter->gain * prior);
	tdl_record(&filter->delay_line, feedback);
	return (filter->gain * feedback) + prior;
}

// Smith-Angell Resonator
struct SAR
{
	float input_samples[2];
	float output_samples[2];
	float a0;
	float a2;
	float b0;
	float b1;
};

static void sar_reset(SAR* filter)
{
	filter->input_samples[0] = 0.0f;
	filter->input_samples[1] = 0.0f;
	filter->output_samples[0] = 0.0f;
	filter->output_samples[1] = 0.0f;
}

static void sar_set_passband(SAR* filter, float center_frequency, int sample_rate, float quality_factor)
{
	float fc = center_frequency;
	float fs = sample_rate;
	float q = quality_factor;

	float theta = tau * fc / fs;
	float w = fc / q;
	float b1 = exp(-tau * w / fs);
	float b0 = -4.0f * b1 / (1.0f + b1) * cos(theta);
	float a0 = 1.0f - sqrt(b1);
	// a1 is zero
	float a2 = -a0;

	filter->a0 = a0;
	filter->a2 = a2;
	filter->b0 = b0;
	filter->b1 = b1;
}

static float sar_apply(SAR* filter, float sample)
{
	// The difference equation is:
	// y(n) = a₀x(n) + a₂x(n-2) - (b₀y(n-1) + b₁y(n-2))
	// where a₀, a₂ are the feedforward coefficients and b₀, b₁ are the feedback
	// coefficients.

	float x0 = sample;
	// since a1 is zero, the x1 term will always be zero
	float x2 = filter->input_samples[1];
	float y0 = filter->output_samples[0];
	float y1 = filter->output_samples[1];
	float a0 = filter->a0;
	float a2 = filter->a2;
	float b0 = filter->b0;
	float b1 = filter->b1;
	float result = (a0 * x0) + (a2 * x2) - ((b0 * y0) + (b1 * y1));

	filter->input_samples[1] = filter->input_samples[0];
	filter->input_samples[0] = sample;

	filter->output_samples[1] = filter->output_samples[0];
	filter->output_samples[0] = result;

	return result;
}

// State-Variable Filter
struct SVF
{
	float prior_low;
	float prior_band;
	float tuning;
	float quality_factor;
};

static void svf_set_center_frequency(SVF* filter, float center_frequency)
{
	filter->tuning = 2.0f * sin(pi * center_frequency);
}

static void svf_set_damping(SVF* filter, float damping)
{
	filter->quality_factor = 2.0f * damping;
}

static void svf_reset(SVF* filter)
{
	filter->prior_low = 0.0f;
	filter->prior_band = 0.0f;
	svf_set_center_frequency(filter, 0.01f);
	svf_set_damping(filter, 1.0f);
}

static float svf_apply(SVF* filter, float sample)
{
	float q = filter->quality_factor;
	float t = filter->tuning;
	float x0 = sample;
	float l1 = filter->prior_low;
	float b1 = filter->prior_band;

	float h0 = x0 - l1 - (q * b1);
	float b0 = (t * h0) + b1;
	float l0 = (t * b0) + l1;

	filter->prior_low = l0;
	filter->prior_band = b0;

	return b0;
}

// §4.5 Effects.................................................................

static const int resonator_filters = 3;

struct Resonator
{
	SAR bank[resonator_filters];
	float gain;
	float mix;
};

static void resonator_default(Resonator* resonator)
{
	for(int i = 0; i < resonator_filters; ++i)
	{
		sar_reset(&resonator->bank[i]);
	}
	resonator->gain = 1.0f;
	resonator->mix = 0.5f;
}

static float resonator_apply(Resonator* resonator, float sample)
{
	float sum = 0.0f;
	for(int i = 0; i < resonator_filters; ++i)
	{
		sum += sar_apply(&resonator->bank[i], sample);
	}
	return lerp(sample, resonator->gain * sum, resonator->mix);
}

struct Overdrive
{
	float factor;
};

static void overdrive_default(Overdrive* overdrive)
{
	overdrive->factor = 1.0f;
}

static float overdrive(float sample, float factor)
{
	// This uses a hyperbola of the formula y = -factor / (x + a) + 1 + a.
	// The quadratic formula is used to find the xy offset, called a, such that
	// the hyperbola passes through the points (0,0) and (1,1). This makes
	// the domain and range both [0,1] for the values being input. Only the
	// magnitude is used so that the sample curves away from zero, and the sign
	// is reintroduced at the end.
	ASSERT(factor > 0.0f);
	factor = fmax(factor, 1e-7f);
	float a = (-1.0f + sqrt(1.0f + 4.0f * factor)) / 2.0f;
	float x = abs(sample);
	float y = -factor / (x + a) + 1.0f + a;
	return copysign(y, sample);
}

struct Distortion
{
	float gain;
	float mix;
};

static void distortion_default(Distortion* distortion)
{
	distortion->gain = 1.0f;
	distortion->mix = 1.0f;
}

static float distort(float sample, float gain, float mix)
{
	if(sample == 0.0f)
	{
		return 0.0f;
	}
	float x = sample;
	float a = x / abs(x);
	float distorted = a * (1.0f - exp(gain * x * a));
	return lerp(sample, distorted, mix);
}

struct RingModulator
{
	float phase_accumulator;
	float phase_step;
	float mix;
	float rate;
};

static void ring_modulator_default(RingModulator* ring_modulator)
{
	ring_modulator->mix = 1.0f;
	ring_modulator->rate = 1.0f;
}

static float ring_modulate(float sample, float phase, float mix)
{
	float modulation = sin(phase);
	modulation = (1.0f - mix + mix * modulation);
	return sample * modulation;
}

static float ring_modulator_apply(RingModulator* ring_modulator, float sample)
{
	ring_modulator->phase_accumulator += ring_modulator->phase_step;
	float phase = ring_modulator->phase_accumulator;
	float mix = ring_modulator->mix;
	float result = ring_modulate(sample, phase, mix);
	ring_modulator->phase_step = tau * ring_modulator->rate;
	return result;
}

struct Bitcrush
{
	float sum;
	float prior;
	int sum_count;
	int depth;
	int downsampling;
};

void bitcrush_reset(Bitcrush* bitcrush)
{
	bitcrush->sum = 0.0f;
	bitcrush->sum_count = 0;
	bitcrush->prior = 0.0f;
}

static float convert_from_s32(s32 x)
{
	return 2.0f / 4294967295.0f * (x + 0.5f);
}

// There's two effects in bitcrushing. One is reducing bit depth. Since the
// output signal has a fixed depth, this is simulated by turning the sample into
// an integer, reducing the depth, and converting back to a floating-point
// sample. The reduction step involves zeroing out the d least significant
// bits of each sample, leaving 32 - d bits.
//
// The second effect is simulating reduced sample rates by taking n samples,
// averaging them, and then outputting that average n times. Note that to get
// n samples to do the averaging involves waiting sample_rate / n seconds, so
// larger amounts of downsampling produces greater latency in the output signal.
// Also, it takes a running sum and computes the average at the end, rather than
// actually keeping all n samples.
float bitcrush_apply(Bitcrush* bitcrush, float sample)
{
	int samples_to_sum = MAX(bitcrush->downsampling, 1);
	if(bitcrush->sum_count < samples_to_sum)
	{
		bitcrush->sum += sample;
		bitcrush->sum_count += 1;
		return bitcrush->prior;
	}

	float average = bitcrush->sum / samples_to_sum;

	float value;
	if(bitcrush->depth < 32)
	{
		int d = 32 - MAX(0, bitcrush->depth);
		s32 s = convert_to_s32(average);
		s &= ~((1 << d) - 1);
		value = convert_from_s32(s);
	}
	else
	{
		value = average;
	}

	bitcrush->prior = value;
	bitcrush->sum = 0.0f;
	bitcrush->sum_count = 0;

	return value;
}

struct Flanger
{
	TDL delay_line;
	float phase_accumulator;
	float phase_step;
	float rate;
	float delay;
	float depth;
	float feedback;
};

static void flanger_create(Flanger* flanger, int max_delay)
{
	tdl_create(&flanger->delay_line, max_delay);
	flanger->phase_accumulator = 0.0f;
	flanger->phase_step = 0.0f;
	flanger->rate = 1.0f;
	flanger->delay = 220.0f;
	flanger->depth = 220.0f;
	flanger->feedback = 0.5f;
}

static void flanger_destroy(Flanger* flanger)
{
	tdl_destroy(&flanger->delay_line);
}

static float flanger_apply(Flanger* flanger, float sample)
{
	flanger->phase_accumulator += flanger->phase_step;
	float modulation = sin(flanger->phase_accumulator);
	flanger->phase_step = flanger->rate * tau;
	modulation = (0.5f * modulation) + 0.5f;

	float delay_time = flanger->depth * modulation + flanger->delay;
	float past_sample = tdl_tap(&flanger->delay_line, delay_time);

	float feedback = sample + flanger->feedback * past_sample;
	tdl_record(&flanger->delay_line, feedback);
	return feedback;
}

static const int phaser_poles = 4;

struct Phaser
{
	TDL delay_lines[phaser_poles];
	float phase_accumulator;
	float phase_step;
	float prior;
	float delay;
	float depth;
	float rate;
	float gain;
	float diffusion;
	float mix;
};

static void phaser_create(Phaser* phaser)
{
	for(int i = 0; i < phaser_poles; ++i)
	{
		tdl_create(&phaser->delay_lines[i], 512);
	}
	phaser->phase_accumulator = 0.0f;
	phaser->phase_step = 0.0f;
	phaser->prior = 0.0f;
	phaser->delay = 32.0f;
	phaser->depth = 16.0f;
	phaser->rate = 0.4f;
	phaser->gain = 0.8f;
	phaser->diffusion = 0.1f;
	phaser->mix = 0.5f;
}

static void phaser_destroy(Phaser* phaser)
{
	for(int i = 0; i < phaser_poles; ++i)
	{
		tdl_destroy(&phaser->delay_lines[i]);
	}
}

static float phaser_apply(Phaser* phaser, float sample)
{
	phaser->phase_accumulator += phaser->phase_step;
	float modulation = sin(phaser->phase_accumulator);
	phaser->phase_step = tau * phaser->rate;
	modulation = (0.5f * modulation) + 0.5f;

	float result = (phaser->gain * phaser->prior) + sample;
	// This is using the delay lines as variable delay all-pass filters.
	for(int i = 0; i < phaser_poles; ++i)
	{
		float delay_time = phaser->depth * modulation + phaser->delay;
		float prior = tdl_tap(&phaser->delay_lines[i], delay_time);
		float feedback = result - (phaser->diffusion * prior);
		tdl_record(&phaser->delay_lines[i], feedback);
		result = (phaser->diffusion * feedback) + prior;
	}
	phaser->prior = result;

	return lerp(sample, result, phaser->mix);
}

struct Vibrato
{
	TDL delay_line;
	float phase_accumulator;
	float phase_step;
	float rate;
	float depth;
	float delay;
};

static void vibrato_create(Vibrato* vibrato)
{
	tdl_create(&vibrato->delay_line, 512);
	vibrato->phase_accumulator = 0.0f;
	vibrato->phase_step = 0.0f;
	vibrato->rate = 8.0f;
	vibrato->depth = 220.0f;
	vibrato->delay = 220.0f;
}

static void vibrato_destroy(Vibrato* vibrato)
{
	tdl_destroy(&vibrato->delay_line);
}

static float vibrato_apply(Vibrato* vibrato, float sample)
{
	vibrato->phase_accumulator += vibrato->phase_step;
	float modulation = sin(vibrato->phase_accumulator);
	vibrato->phase_step = tau * vibrato->rate;
	modulation = (0.5f * modulation) + 0.5f;
	float delay_time = vibrato->depth * modulation + vibrato->delay;

	float result = tdl_tap(&vibrato->delay_line, delay_time);
	tdl_record(&vibrato->delay_line, sample);
	return result;
}

struct Chorus
{
	TDL delay_line;
	float phase_accumulator;
	float phase_step;
	float rate;
	float depth;
	float delay;
};

static void chorus_create(Chorus* chorus)
{
	tdl_create(&chorus->delay_line, 2048);
	chorus->phase_accumulator = 0.0f;
	chorus->phase_step = 0.0f;
	chorus->delay = 1024.0f;
	chorus->depth = 40.0f;
	chorus->rate = 2.0f;
}

static void chorus_destroy(Chorus* chorus)
{
	tdl_destroy(&chorus->delay_line);
}

static float chorus_apply(Chorus* chorus, float sample)
{
	chorus->phase_accumulator += chorus->phase_step;
	float modulation = sin(chorus->phase_accumulator);
	chorus->phase_step = tau * chorus->rate;
	modulation = (0.5f * modulation) + 0.5f;
	float delay_time = chorus->depth * modulation + chorus->delay;
	float result = sample + tdl_tap(&chorus->delay_line, delay_time);
	tdl_record(&chorus->delay_line, sample);
	return result;
}

struct Delay
{
	TDL delay_line;
	float delay;
	float feedback;
	float mix;
};

static void delay_create(Delay* delay)
{
	tdl_create(&delay->delay_line, 65536);
	delay->delay = 100.0f;
	delay->feedback = 0.3f;
	delay->mix = 0.5f;
}

static void delay_destroy(Delay* delay)
{
	tdl_destroy(&delay->delay_line);
}

static float delay_apply(Delay* delay, float sample)
{
	float past_sample = tdl_tap(&delay->delay_line, delay->delay);
	float feedback = sample + delay->feedback * past_sample;
	tdl_record(&delay->delay_line, feedback);
	float result = lerp(sample, feedback, delay->mix);
	return result;
}

struct AutoWah
{
	SVF filter;
	float phase_accumulator;
	float rate;
	float mix;
	float depth;
	float breadth;
};

static void auto_wah_default(AutoWah* wah)
{
	svf_reset(&wah->filter);
	wah->phase_accumulator = 0.0f;
	wah->rate = 0.001f;
	wah->mix = 0.8f;
	wah->breadth = 0.03f;
	wah->depth = 0.01f;
}

static float auto_wah_apply(AutoWah* wah, float sample)
{
	wah->phase_accumulator += wah->rate;
	float phi = triangle_wave(wah->phase_accumulator);
	float frequency = (wah->breadth * phi) + wah->depth + wah->breadth;
	svf_set_center_frequency(&wah->filter, frequency);
	float filtered = svf_apply(&wah->filter, sample);
	return lerp(sample, filtered, wah->mix);
}

// §4.6 Voice...................................................................

struct Voice
{
	ADSR amp_envelope;
	ADSR pitch_envelope;
	arandom::Sequence randomness;
	float phase_accumulator;
	float phase_step;
	float fm_phase_accumulator;
	float fm_phase_step;
};

static bool voice_is_unused(Voice* voice)
{
	return voice->amp_envelope.state == ADSR::State::Neutral;
}

static void voice_gate(Voice* voice, bool gate, bool use_pitch_envelope)
{
	envelope_gate(&voice->amp_envelope, gate);
	if(use_pitch_envelope)
	{
		envelope_gate(&voice->pitch_envelope, gate);
	}
}

static void voice_reset(Voice* voice)
{
	envelope_reset(&voice->amp_envelope);
	envelope_reset(&voice->pitch_envelope);
	arandom::seed_by_time(&voice->randomness);
	voice->phase_accumulator = 0.0f;
	voice->phase_step = 0.0f;
	voice->fm_phase_accumulator = 0.0f;
	voice->fm_phase_step = 0.0f;
}

// §4.7 Voice Map...............................................................

struct VoiceEntry
{
	u32 track : 3;
	u32 note : 29;
};

static const u32 no_note = 0x1fffffff;
static const u32 no_track = 0x7;

static void voice_map_setup(VoiceEntry* voice_map, int count)
{
	for(int i = 0; i < count; ++i)
	{
		voice_map[i].track = no_track;
		voice_map[i].note = no_note;
	}
}

static bool find_voice(VoiceEntry* voice_map, int voices, int track, int note, int* voice)
{
	for(int i = 0; i < voices; ++i)
	{
		if(voice_map[i].track == track && voice_map[i].note == note)
		{
			*voice = i;
			return true;
		}
	}
	return false;
}

static int assign_voice(VoiceEntry* voice_map, int voices, int track, int note)
{
	for(int i = 0; i < voices; ++i)
	{
		if(voice_map[i].note == no_note)
		{
			voice_map[i].track = track;
			voice_map[i].note = note;
			return i;
		}
	}
	// If all voices are currently in use, evict the first one.
	voice_map[0].track = track;
	voice_map[0].note = note;
	return 0;
}

static void free_associated_voices(VoiceEntry* voice_map, int voices, int track)
{
	for(int i = 0; i < voices; ++i)
	{
		if(voice_map[i].track == track)
		{
			voice_map[i].track = no_track;
			voice_map[i].note = no_note;
		}
	}
}

// History......................................................................

static const int history_events_cap = 32;

struct History
{
	struct Event
	{
		double time;
		int track;
		bool started;
	};

	Event events[history_events_cap];
	int count;
};

static void history_erase(History* history)
{
	history->count = 0;
}

static void history_record(History* history, int track_index, double time, bool started)
{
	History::Event* record = &history->events[history->count];
	record->time = time;
	record->track = track_index;
	record->started = started;
	ASSERT(history->count + 1 < history_events_cap);
	history->count = (history->count + 1) % history_events_cap;
}

// Envelope Settings............................................................

struct EnvelopeSettings
{
	struct
	{
		float attack;
		float decay;
		float sustain;
		float release;
		int semitones;
		bool use;
	} pitch;

	struct
	{
		float attack;
		float decay;
		float sustain;
		float release;
	} amp;
};

// §4.8 Track...................................................................

#define C  0
#define CS 1
#define D  2
#define DS 3
#define E  4
#define F  5
#define FS 6
#define G  7
#define GS 8
#define A  9
#define AS 10
#define B  11

namespace
{
	int pentatonic_major[] = {4, 2, 4, 7, 9};
	int pentatonic_minor[] = {4, 3, 5, 7, 9};
}

#undef C
#undef CS
#undef D
#undef DS
#undef E
#undef F
#undef FS
#undef G
#undef GS
#undef A
#undef AS
#undef B

enum class Section
{
	Intro,
	Verse,
	Chorus,
	Bridge,
	Breakdown,
	Coda,
};

struct Composer
{
	struct State
	{
		Section section;
		int next_state;
	};

	State states[16];
	int state;
};

static void compose_form(Composer* composer)
{
	composer->states[0].section = Section::Verse;
	composer->states[0].next_state = 1;

	composer->states[1].section = Section::Chorus;
	composer->states[1].next_state = 0;
}

static void composer_update_state(Composer* composer)
{
	Composer::State* state = &composer->states[composer->state];
	composer->state = state->next_state;
}

struct Track
{
	struct Note
	{
		int pitch;
		float velocity;
	};

	struct Event
	{
		double time;
		int note;
	};

	Note notes[32];
	Event start_events[32];
	Event stop_events[32];
	int notes_count;
	int notes_capacity;
	int start_index;
	int stop_index;

	int octave;
	int octave_range;
	int style;
};

#define COMPARE_TIME(a, b)\
	a.time > b.time

DEFINE_INSERTION_SORT(Track::Event, COMPARE_TIME, by_time);

void track_setup(Track* track)
{
	track->notes_count = 0;
	track->notes_capacity = 32;
	track->start_index = 0;
	track->stop_index = 0;
	track->octave = 1;
	track->octave_range = 1;
	track->style = 0;
}

void track_generate(Track* track, double finish_time, Composer* composer, arandom::Sequence* randomness)
{
	Composer::State* state = &composer->states[composer->state];

	double note_spacing;
	double note_length;
	switch(state->section)
	{
		default:
		case Section::Verse:
		{
			switch(track->style)
			{
				case 0:
				{
					note_spacing = 0.6;
					note_length = 0.2;
					break;
				}
				case 1:
				{
					note_spacing = 1.2;
					note_length = 0.08;
					break;
				}
				case 2:
				{
					note_spacing = 0.3;
					note_length = 0.3;
					break;
				}
				case 3:
				{
					note_spacing = 0.3;
					note_length = 0.08;
					break;
				}
			}
			break;
		}
		case Section::Chorus:
		{
			switch(track->style)
			{
				case 0:
				{
					note_spacing = 0.6;
					note_length = 0.1;
					break;
				}
				case 1:
				{
					note_spacing = 0.6;
					note_length = 0.08;
					break;
				}
				case 2:
				{
					note_spacing = 1.2;
					note_length = 0.6;
					break;
				}
				case 3:
				{
					note_spacing = 0.2;
					note_length = 0.08;
					break;
				}
			}
			break;
		}
	}

	int generated = 0;
	for(int i = track->notes_count; i < track->notes_capacity; ++i)
	{
		double note_start = note_spacing * generated;

		if(note_start >= finish_time)
		{
			break;
		}

		int degrees = pentatonic_major[0];
		int degree = arandom::int_range(randomness, 0, degrees - 1);
		int pitch_class = pentatonic_major[degree + 1];

		int lowest_octave = track->octave;
		int highest_octave = track->octave + track->octave_range;
		int octave = arandom::int_range(randomness, lowest_octave, highest_octave);

		Track::Note* note = &track->notes[i];
		note->pitch = 12 * octave + pitch_class;
		note->velocity = 1.0f;

		Track::Event* start_event = &track->start_events[i];
		start_event->note = i;
		start_event->time = note_start;

		Track::Event* stop_event = &track->stop_events[i];
		stop_event->note = i;
		stop_event->time = note_start + note_length;

		generated += 1;
	}

	if(track->notes_count == track->notes_capacity)
	{
		LOG_DEBUG("The audio track note capacity was exceeded.");
	}

	track->notes_count = MIN(track->notes_count + generated, track->notes_capacity);

	insertion_sort_by_time(track->start_events, track->notes_count);
	insertion_sort_by_time(track->stop_events, track->notes_count);
}

static void transfer_unfinished_notes(Track* track, double time)
{
	const int transfer_cap = 8;
	Track::Note notes_to_transfer[transfer_cap];
	Track::Event stops_to_transfer[transfer_cap];
	int transferred = 0;
	for(int i = track->stop_index; i < track->notes_count && transferred < transfer_cap; ++i)
	{
		Track::Event stop = track->stop_events[i];
		stop.time -= time;
		stops_to_transfer[transferred] = stop;
		notes_to_transfer[transferred] = track->notes[stop.note];
		transferred += 1;
	}

	for(int i = 0; i < transferred; ++i)
	{
		track->notes[i] = notes_to_transfer[i];
		track->stop_events[i] = stops_to_transfer[i];

		track->start_events[i].note = 0;
		// Make sure these sort to the beginning of the list by using a negative
		// time value.
		track->start_events[i].time = -1.0;
	}

	track->notes_count = transferred;
	track->start_index = transferred;
	track->stop_index = 0;
}

static bool should_regenerate(Track* track)
{
	return track->start_index == track->notes_count;
}

struct RenderResult
{
	int voices[16];
	float pitches[16];
	int voices_count;
};

void track_render(Track* track, int track_index, Voice* voices, VoiceEntry* voice_map, int count, double time, EnvelopeSettings* envelope_settings, History* history, RenderResult* result)
{
	// Gate any start events in this time slice.
	for(int i = track->start_index; i < track->notes_count; ++i)
	{
		Track::Event* start = &track->start_events[i];
		if(start->time > time)
		{
			// This event is in the future, so all events after it are also.
			break;
		}
		else
		{
			ASSERT(start->time >= 0.0f);
			int index;
			bool found = find_voice(voice_map, count, track_index, start->note, &index);
			if(!found)
			{
				index = assign_voice(voice_map, count, track_index, start->note);
				Voice* voice = &voices[index];
				voice_reset(voice);
				envelope_set_attack(&voice->amp_envelope, envelope_settings->amp.attack);
				envelope_set_decay(&voice->amp_envelope, envelope_settings->amp.decay);
				envelope_set_sustain(&voice->amp_envelope, envelope_settings->amp.sustain);
				envelope_set_release(&voice->amp_envelope, envelope_settings->amp.release);
				envelope_set_attack(&voice->pitch_envelope, envelope_settings->pitch.attack);
				envelope_set_decay(&voice->pitch_envelope, envelope_settings->pitch.decay);
				envelope_set_sustain(&voice->pitch_envelope, envelope_settings->pitch.sustain);
				envelope_set_release(&voice->pitch_envelope, envelope_settings->pitch.release);
				voice_gate(voice, true, envelope_settings->pitch.use);
			}
			history_record(history, track_index, start->time, true);
			track->start_index = i + 1;
		}
	}

	// Gate any stop events.
	for(int i = track->stop_index; i < track->notes_count; ++i)
	{
		Track::Event* stop = &track->stop_events[i];
		if(stop->time > time)
		{
			// This event is in the future, so all events after it are also.
			break;
		}
		else
		{
			ASSERT(stop->time >= 0.0f);
			int index;
			bool found = find_voice(voice_map, count, track_index, stop->note, &index);
			if(found)
			{
				voice_gate(&voices[index], false, envelope_settings->pitch.use);
			}
			history_record(history, track_index, stop->time, false);
			track->stop_index = i + 1;
		}
	}

	// Fill the result given the new status of the voices. In the same cycle,
	// free up any voices that have finished sounding out their note.
	for(int i = 0; i < count; ++i)
	{
		if(voice_map[i].track != track_index)
		{
			continue;
		}
		int note_index = voice_map[i].note;
		if(note_index != no_note)
		{
			if(voice_is_unused(&voices[i]))
			{
				voice_map[i].track = no_track;
				voice_map[i].note = no_note;
			}
			else
			{
				Track::Note note = track->notes[note_index];
				result->voices[result->voices_count] = i;
				result->pitches[result->voices_count] = note.pitch;
				result->voices_count += 1;
			}
		}
	}
}

// §4.9 Stream..................................................................

struct Stream
{
	float* samples;
	int samples_count;
	int channels;
	float volume;
	float pan;
};

static void stream_create(Stream* stream, int capacity, int channels)
{
	stream->samples = ALLOCATE(float, capacity);
	stream->samples_count = capacity;
	stream->volume = 1.0f;
	stream->channels = 1;
	stream->pan = 0.0f;
}

static void stream_destroy(Stream* stream)
{
	if(stream)
	{
		SAFE_DEALLOCATE(stream->samples);
	}
}

static void mix_streams(Stream* streams, int streams_count, float* mixed_samples, int frames, int channels, float volume)
{
	int samples = frames * channels;
	fill_with_silence(mixed_samples, 0, samples);
	for(int i = 0; i < streams_count; ++i)
	{
		Stream* stream = streams + i;
		if(channels < stream->channels)
		{
			for(int j = 0; j < frames; ++j)
			{
				for(int k = 0; k < channels; ++k)
				{
					mixed_samples[channels * j + k] += stream->volume * stream->samples[stream->channels * j];
				}
			}
		}
		else if(channels > stream->channels)
		{
			// This only handles mixing monaural to multiple channels, not
			// stereo-to-surround mixing.
			ASSERT(stream->channels == 1);
			if(channels == 2)
			{
				float theta = (0.5f * -stream->pan + 0.5f) * pi_over_2;
				float pan_left = sin(theta);
				float pan_right = cos(theta);
				for(int j = 0; j < frames; ++j)
				{
					float sample = stream->volume * stream->samples[stream->channels * j];
					mixed_samples[channels * j] += pan_left * sample;
					mixed_samples[channels * j + 1] += pan_right * sample;
				}
			}
			else
			{
				for(int j = 0; j < frames; ++j)
				{
					float sample = stream->volume * stream->samples[stream->channels * j];
					for(int k = 0; k < channels; ++k)
					{
						mixed_samples[channels * j + k] += sample;
					}
				}
			}
		}
		else
		{
			for(int j = 0; j < samples; ++j)
			{
				mixed_samples[j] += stream->volume * stream->samples[j];
			}
		}
	}
	for(int i = 0; i < samples; ++i)
	{
		mixed_samples[i] = clamp(volume * mixed_samples[i], -1.0f, 1.0f);
	}
}

// §4.10 Message Queue..........................................................

struct Message
{
	enum class Code
	{
		Boop,
		Note,
		Oscilloscope_Settings,
		Oscilloscope_Channel,
		Oscilloscope_Samples,
	} code;

	union
	{
		struct
		{
			bool on;
		} boop;

		struct
		{
			int track;
			bool start;
		} note;

		struct
		{
			int sample_rate;
		} oscilloscope_settings;

		struct
		{
			int index;
			bool active;
		} oscilloscope_channel;

		struct
		{
			float* array;
			int frames;
			int channels;
		} oscilloscope_samples;
	};
};

static const int messages_cap = 32;

struct MessageQueue
{
	Message messages[messages_cap];
	AtomicInt head;
	AtomicInt tail;
};

static bool was_empty(MessageQueue* queue)
{
	return atomic_int_load(&queue->head) == atomic_int_load(&queue->tail);
}

static bool was_full(MessageQueue* queue)
{
	int next_tail = atomic_int_load(&queue->tail);
	next_tail = (next_tail + 1) % messages_cap;
	return next_tail == atomic_int_load(&queue->head);
}

static bool enqueue_message(MessageQueue* queue, Message* message)
{
	int current_tail = atomic_int_load(&queue->tail);
	int next_tail = (current_tail + 1) % messages_cap;
	if(next_tail != atomic_int_load(&queue->head))
	{
		queue->messages[current_tail] = *message;
		atomic_int_store(&queue->tail, next_tail);
		return true;
	}
	return false;
}

static bool dequeue_message(MessageQueue* queue, Message* message)
{
	int current_head = atomic_int_load(&queue->head);
	if(current_head == atomic_int_load(&queue->tail))
	{
		return false;
	}
	*message = queue->messages[current_head];
	atomic_int_store(&queue->head, (current_head + 1) % messages_cap);
	return true;
}

// Effect.......................................................................

enum class EffectType
{
	Low_Pass_Filter,
	Band_Pass_Filter,
	High_Pass_Filter,
	Reverb,
	Resonator,
	Overdrive,
	Distortion,
	Ring_Modulator,
	Bitcrush,
	Flanger,
	Phaser,
	Vibrato,
	Chorus,
	Delay,
	Auto_Wah,
};

struct Effect
{
	union
	{
		LPF lowpass;
		BPF bandpass;
		HPF highpass;
		APF reverb;
		Resonator resonator;
		Overdrive overdrive;
		Distortion distortion;
		RingModulator ring_modulator;
		Bitcrush bitcrush;
		Flanger flanger;
		Phaser phaser;
		Vibrato vibrato;
		Chorus chorus;
		Delay delay;
		AutoWah auto_wah;
	};
	EffectType type;
};

static void apply_effects(Effect* effects, int count, Stream* stream)
{
	int frames = stream->samples_count / stream->channels;
	for(int i = 0; i < count; ++i)
	{
		Effect* effect = &effects[i];
		switch(effect->type)
		{
			case EffectType::Low_Pass_Filter:
			{
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = lpf_apply(&effect->lowpass, value);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
			case EffectType::Band_Pass_Filter:
			{
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = bpf_apply(&effect->bandpass, value);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
			case EffectType::High_Pass_Filter:
			{
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = hpf_apply(&effect->highpass, value);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
			case EffectType::Reverb:
			{
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = apf_apply(&effect->reverb, value);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
			case EffectType::Resonator:
			{
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = resonator_apply(&effect->resonator, value);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
			case EffectType::Overdrive:
			{
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = overdrive(value, effect->overdrive.factor);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
			case EffectType::Distortion:
			{
				float gain = effect->distortion.gain;
				float mix = effect->distortion.mix;
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = distort(value, gain, mix);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
			case EffectType::Ring_Modulator:
			{
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = ring_modulator_apply(&effect->ring_modulator, value);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
			case EffectType::Bitcrush:
			{
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = bitcrush_apply(&effect->bitcrush, value);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
			case EffectType::Flanger:
			{
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = flanger_apply(&effect->flanger, value);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
			case EffectType::Phaser:
			{
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = phaser_apply(&effect->phaser, value);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
			case EffectType::Vibrato:
			{
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = vibrato_apply(&effect->vibrato, value);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
			case EffectType::Chorus:
			{
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = chorus_apply(&effect->chorus, value);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
			case EffectType::Delay:
			{
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = delay_apply(&effect->delay, value);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
			case EffectType::Auto_Wah:
			{
				for(int j = 0; j < frames; ++j)
				{
					float value = stream->samples[stream->channels * j];
					value = auto_wah_apply(&effect->auto_wah, value);
					for(int k = 0; k < stream->channels; ++k)
					{
						stream->samples[stream->channels * j + k] = value;
					}
				}
				break;
			}
		}
	}
}

// §4.11 Generate Oscillation...................................................

enum class Oscillator
{
	Sine,
	Square,
	Pulse,
	Sawtooth,
	Triangle,
	Rectified_Sine,
	Cycloid,
	Noise,
	FM_Sine,
};

static const int instrument_effects_cap = 8;

struct Instrument
{
	Effect effects[instrument_effects_cap];
	EnvelopeSettings envelope_settings;

	union
	{
		struct
		{
			float width;
		} pulse;
		struct
		{
			BPF filter;
			int passband;
		} noise;
		struct
		{
			float ratio;
			float gain;
		} fm; // frequency modulation
	};

	Oscillator oscillator;
	int effects_count;
	float pulse_width;
};

static void instrument_destroy(Instrument* instrument)
{
	for(int i = 0; i < instrument->effects_count; ++i)
	{
		Effect* effect = &instrument->effects[i];
		switch(effect->type)
		{
			case EffectType::Reverb:
			{
				apf_destroy(&effect->reverb);
				break;
			}
			case EffectType::Flanger:
			{
				flanger_destroy(&effect->flanger);
				break;
			}
			case EffectType::Phaser:
			{
				phaser_destroy(&effect->phaser);
				break;
			}
			case EffectType::Vibrato:
			{
				vibrato_destroy(&effect->vibrato);
				break;
			}
			case EffectType::Chorus:
			{
				chorus_destroy(&effect->chorus);
				break;
			}
			case EffectType::Delay:
			{
				delay_destroy(&effect->delay);
				break;
			}
			default:
			{
				// All other effects have no need for a destroy routine.
				break;
			}
		}
	}
}

static Effect* instrument_add_effect(Instrument* instrument, EffectType type)
{
	int index = instrument->effects_count;
	ASSERT(instrument->effects_count + 1 < instrument_effects_cap);
	instrument->effects_count = (index + 1) % instrument_effects_cap;
	Effect* effect = &instrument->effects[index];
	effect->type = type;

	switch(type)
	{
		case EffectType::Low_Pass_Filter:
		{
			lpf_reset(&effect->lowpass);
			break;
		}
		case EffectType::High_Pass_Filter:
		{
			hpf_reset(&effect->highpass);
			break;
		}
		case EffectType::Band_Pass_Filter:
		{
			bpf_reset(&effect->bandpass);
			break;
		}
		case EffectType::Reverb:
		{
			apf_create(&effect->reverb, 4096);
			break;
		}
		case EffectType::Resonator:
		{
			resonator_default(&effect->resonator);
			break;
		}
		case EffectType::Overdrive:
		{
			overdrive_default(&effect->overdrive);
			break;
		}
		case EffectType::Distortion:
		{
			distortion_default(&effect->distortion);
			break;
		}
		case EffectType::Ring_Modulator:
		{
			ring_modulator_default(&effect->ring_modulator);
			break;
		}
		case EffectType::Bitcrush:
		{
			bitcrush_reset(&effect->bitcrush);
			break;
		}
		case EffectType::Flanger:
		{
			flanger_create(&effect->flanger, 2048);
			break;
		}
		case EffectType::Phaser:
		{
			phaser_create(&effect->phaser);
			break;
		}
		case EffectType::Vibrato:
		{
			vibrato_create(&effect->vibrato);
			break;
		}
		case EffectType::Chorus:
		{
			chorus_create(&effect->chorus);
			break;
		}
		case EffectType::Delay:
		{
			delay_create(&effect->delay);
			break;
		}
		case EffectType::Auto_Wah:
		{
			auto_wah_default(&effect->auto_wah);
			break;
		}
	}

	return effect;
}

static float process_sine(Voice* voice, Instrument* instrument, float theta, int sample_rate)
{
	voice->phase_step = tau * theta / sample_rate;
	return sin(voice->phase_accumulator);
}

static float process_square(Voice* voice, Instrument* instrument, float theta, int sample_rate)
{
	voice->phase_step = theta / sample_rate;
	return square_wave(voice->phase_accumulator);
}

static float process_pulse(Voice* voice, Instrument* instrument, float theta, int sample_rate)
{
	voice->phase_step = theta / sample_rate;
	return pulse_wave(voice->phase_accumulator, instrument->pulse.width);
}

static float process_triangle(Voice* voice, Instrument* instrument, float theta, int sample_rate)
{
	voice->phase_step = theta / sample_rate;
	return triangle_wave(voice->phase_accumulator);
}

static float process_sawtooth(Voice* voice, Instrument* instrument, float theta, int sample_rate)
{
	voice->phase_step = theta / sample_rate;
	return sawtooth_wave(voice->phase_accumulator);
}

static float process_rectified_sine(Voice* voice, Instrument* instrument, float theta, int sample_rate)
{
	voice->phase_step = tau * theta / sample_rate;
	return rectified_sin(voice->phase_accumulator);
}

static float process_cycloid(Voice* voice, Instrument* instrument, float theta, int sample_rate)
{
	voice->phase_step = tau * theta / sample_rate;
	return cycloid(voice->phase_accumulator);
}

static float process_noise(Voice* voice, Instrument* instrument, float theta, int sample_rate)
{
	// Since white noise is uniform across the frequency spectrum, it can't be
	// pitched as is, so a band-pass filter is used.
	int passband_extent = instrument->noise.passband;
	float delta_time = 1.0f / sample_rate;
	float low = fmax(frequency_given_interval(theta, -passband_extent), 0.01f);
	float high = frequency_given_interval(theta, passband_extent);
	BPF* filter = &instrument->noise.filter;
	bpf_set_passband(filter, low, high, delta_time);
	float value = arandom::float_range(&voice->randomness, -1.0f, 1.0f);
	value = bpf_apply(filter, value);
	return value;
}

static float process_fm_sine(Voice* voice, Instrument* instrument, float theta, int sample_rate)
{
	float ratio = instrument->fm.ratio;
	float gain = instrument->fm.gain;
	voice->phase_step = tau * theta / sample_rate;
	float fm_theta = ratio * theta;
	voice->fm_phase_accumulator += voice->fm_phase_step;
	voice->fm_phase_step = tau * fm_theta / sample_rate;
	float phase = (gain / fm_theta) * (sin(voice->fm_phase_accumulator - pi_over_2) + 1.0f);
	return sin(voice->phase_accumulator + phase);
}

static void generate_oscillation(Stream* stream, int start_frame, int end_frame, Track* track, int track_index, Instrument* instrument, Voice* voices, VoiceEntry* voice_map, int voices_count, int sample_rate, double time, History* history)
{
	// These are some pretty absurd macros. Basically, most of the oscillators
	// have identical loops, but the function called to generate the next sample
	// is different.
	//
	// Also, to prevent having to check whether the pitch envelope is needed for
	// every frame, it's easiest to split that into its own loop and just check
	// once and enter the appropriate loop.

#define JUST_OSCILLATOR(process)\
	for(int i = start_frame; i < end_frame; ++i)\
	{\
		double t = static_cast<double>(i - start_frame) / sample_rate + time;\
		RenderResult result = {};\
		track_render(track, track_index, voices, voice_map, voices_count, t, envelope_settings, history, &result);\
		for(int j = 0; j < result.voices_count; ++j)\
		{\
			Voice* voice = &voices[result.voices[j]];\
			int pitch = result.pitches[j];\
			float theta = pitch_to_frequency(pitch);\
			voice->phase_accumulator += voice->phase_step;\
			float value = process(voice, instrument, theta, sample_rate);\
			value = envelope_apply(&voice->amp_envelope) * value;\
			for(int k = 0; k < stream->channels; ++k)\
			{\
				stream->samples[stream->channels * i + k] += value;\
			}\
		}\
	}

#define OSCILLATOR_WITH_PITCH_ENVELOPE(process)\
	for(int i = start_frame; i < end_frame; ++i)\
	{\
		double t = static_cast<double>(i - start_frame) / sample_rate + time;\
		RenderResult result = {};\
		track_render(track, track_index, voices, voice_map, voices_count, t, envelope_settings, history, &result);\
		for(int j = 0; j < result.voices_count; ++j)\
		{\
			Voice* voice = &voices[result.voices[j]];\
			int pitch = result.pitches[j];\
			float unpitched_theta = pitch_to_frequency(pitch);\
			float pitched_theta = pitch_to_frequency(pitch + semitones);\
			float modulation = envelope_apply(&voice->pitch_envelope);\
			float theta = lerp(unpitched_theta, pitched_theta, modulation);\
			voice->phase_accumulator += voice->phase_step;\
			float value = process(voice, instrument, theta, sample_rate);\
			value = envelope_apply(&voice->amp_envelope) * value;\
			for(int k = 0; k < stream->channels; ++k)\
			{\
				stream->samples[stream->channels * i + k] += value;\
			}\
		}\
	}

	EnvelopeSettings* envelope_settings = &instrument->envelope_settings;
	int semitones = envelope_settings->pitch.semitones;
	bool use_pitch_envelope = envelope_settings->pitch.use;

	switch(instrument->oscillator)
	{
		case Oscillator::Sine:
		{
			if(use_pitch_envelope)
			{
				OSCILLATOR_WITH_PITCH_ENVELOPE(process_sine);
			}
			else
			{
				JUST_OSCILLATOR(process_sine);
			}
			break;
		}
		case Oscillator::Square:
		{
			if(use_pitch_envelope)
			{
				OSCILLATOR_WITH_PITCH_ENVELOPE(process_square);
			}
			else
			{
				JUST_OSCILLATOR(process_square);
			}
			break;
		}
		case Oscillator::Pulse:
		{
			if(use_pitch_envelope)
			{
				OSCILLATOR_WITH_PITCH_ENVELOPE(process_pulse);
			}
			else
			{
				JUST_OSCILLATOR(process_pulse);
			}
			break;
		}
		case Oscillator::Triangle:
		{
			if(use_pitch_envelope)
			{
				OSCILLATOR_WITH_PITCH_ENVELOPE(process_triangle);
			}
			else
			{
				JUST_OSCILLATOR(process_triangle);
			}
			break;
		}
		case Oscillator::Sawtooth:
		{
			if(use_pitch_envelope)
			{
				OSCILLATOR_WITH_PITCH_ENVELOPE(process_sawtooth);
			}
			else
			{
				JUST_OSCILLATOR(process_sawtooth);
			}
			break;
		}
		case Oscillator::Rectified_Sine:
		{
			if(use_pitch_envelope)
			{
				OSCILLATOR_WITH_PITCH_ENVELOPE(process_rectified_sine);
			}
			else
			{
				JUST_OSCILLATOR(process_rectified_sine);
			}
			break;
		}
		case Oscillator::Cycloid:
		{
			if(use_pitch_envelope)
			{
				OSCILLATOR_WITH_PITCH_ENVELOPE(process_cycloid);
			}
			else
			{
				JUST_OSCILLATOR(process_cycloid);
			}
			break;
		}
		case Oscillator::Noise:
		{
			bpf_reset(&instrument->noise.filter);
			if(use_pitch_envelope)
			{
				OSCILLATOR_WITH_PITCH_ENVELOPE(process_noise);
			}
			else
			{
				JUST_OSCILLATOR(process_noise);
			}
			break;
		}
		case Oscillator::FM_Sine:
		{
			if(use_pitch_envelope)
			{
				OSCILLATOR_WITH_PITCH_ENVELOPE(process_fm_sine);
			}
			else
			{
				JUST_OSCILLATOR(process_fm_sine);
			}
			break;
		}
	}
#undef OSCILLATOR_WITH_PITCH_ENVELOPE
#undef JUST_OSCILLATOR
}

} // namespace audio

// Waveform Similarity based Overlap-Add..........................................

static float hann_window(int i, int count)
{
	return 0.5f * (1.0f - cos(tau * i / static_cast<float>(count)));
}

static int find_greatest_cross_correlation(float* s0, int s0_count, float* s1, int s1_count, int max_lag)
{
	int max_index = 0;
	float max = -infinity;
	for(int i = -max_lag; i <= max_lag; ++i)
	{
		float sum = 0.0f;
		for(int j0 = 0; j0 < s0_count; ++j0)
		{
			int j1 = j0 + i;
			if(j1 >= 0 && j1 < s1_count)
			{
				sum += s0[j0] * s1[j1];
			}
		}
		// The normalized cross correlation would be the sum at this point
		// divided by the square root of the product of the sums of s0 and s1.
		// Like: √(Σs₀×Σs₁) But only the index of the maximum correlation is
		// needed and not its actual value, so the normalization is skipped.
		if(sum > max)
		{
			max = sum;
			max_index = i + max_lag;
		}
	}
	return max_index;
}

struct WSOLA
{
	float* output;
	float* window_effect;
	int output_count;
	int analysis_shift;
	int analysis_count;
	int deltas;
	float scale_factor;
};

static void wsola_create(WSOLA* dilator)
{
	dilator->output = nullptr;
	dilator->window_effect = nullptr;
	dilator->output_count = 0;
	dilator->analysis_shift = 256;
	dilator->analysis_count = 1024;
	dilator->deltas = 128;
	dilator->scale_factor = 2.0f;
}

static void wsola_destroy(WSOLA* dilator)
{
	if(dilator)
	{
		SAFE_DEALLOCATE(dilator->output);
		SAFE_DEALLOCATE(dilator->window_effect);
	}
}

static void dilate_time(WSOLA* dilator, float* samples, int samples_count)
{
	int analysis_shift = dilator->analysis_shift;
	int analysis_count = dilator->analysis_count;
	int deltas = dilator->deltas;
	float scale_factor = dilator->scale_factor;

	int dilated_count = floor(samples_count / scale_factor + analysis_count + 0.5f);
	float* result = REALLOCATE(dilator->output, float, dilated_count);
	dilator->output = result;
	dilator->output_count = dilated_count;
	float* window_effect = REALLOCATE(dilator->window_effect, float, dilated_count);
	dilator->window_effect = window_effect;

	int index0 = analysis_shift;
	int index1;
	int alpha = 0;
	int result_index = 0;

	while(index0 + analysis_count < samples_count && alpha + analysis_count + deltas + scale_factor * analysis_shift < samples_count)
	{
		// Determine the bounds of two segments of the signal to analyse.
		alpha += round(scale_factor * analysis_shift);
		index1 = MAX(alpha - deltas, 0);
		int segment1_count = (alpha + analysis_count - 1 + deltas) - index1;
		result_index += analysis_shift;

		// Find the matching point in the two segments of the signal.
		int max_index = find_greatest_cross_correlation(&samples[index0], analysis_count, &samples[index1], segment1_count, deltas);

		// Overlap and add the result samples, using the matching point to
		// position it. Also, keep a window normalization record for later.
		int offset = alpha - deltas + max_index;
		for(int i = 0; i < analysis_count; ++i)
		{
			float window_factor = hann_window(i, analysis_count);
			result[result_index + i] += window_factor * samples[offset + i];
			window_effect[result_index + i] += window_factor;
		}

		// Also set up the next analysis phase relative to the matching point.
		index0 = offset + analysis_shift;
	}

	// The windowing process causes amplitude fluctuation in the output that
	// needs to be normalised out. The epsilon just prevents dividing by zero
	// in places where succeeding windows may not overlap and leave a space of
	// non-windowed silence.
	for(int i = 0; i < dilated_count; ++i)
	{
		result[i] /= window_effect[i] + FLT_EPSILON;
	}
}

static void resample_linear(float* samples, int samples_count, float* result, int result_count)
{
	float ratio = (samples_count - 1) / static_cast<float>(result_count);
	for(int i = 0; i < result_count; ++i)
	{
		float x = i * ratio;
		int xi = floor(x);
		float xf = x - xi;
		result[i] = lerp(samples[xi], samples[xi + 1], xf);
	}
}

static void shift_pitch(WSOLA* dilator, float* samples, int samples_count, int semitones)
{
	dilator->scale_factor = 1.0f / pow(2.0f, semitones / 12.0f);
	dilate_time(dilator, samples, samples_count);
	resample_linear(dilator->output, dilator->output_count, samples, samples_count);
}

// §4.12 Audio System Declarations..............................................

namespace audio {

bool system_startup();
void system_shutdown();
void system_send_message(Message* message);

} // namespace audio

// §5.1 Game....................................................................

namespace
{
	const char* app_name = "ONE";
	const int window_width = 800;
	const int window_height = 600;
	const double frame_frequency = 1.0 / 60.0;

	Vector3 position;
	World world;
	Vector3 dancer_position;
	audio::MessageQueue message_queue;

	profile::Inspector profile_inspector;
}

static void game_create()
{
	Collider* collider = world.colliders;
	world.colliders_count += 1;
	collider->triangles = render::terrain_triangles;
	collider->triangles_count = render::terrain_triangles_count;
	bool built = bih::build_tree(&collider->tree, collider->triangles, collider->triangles_count);
	ASSERT(built);

	// Setup the tweaker.
	{
		float leading = 34.0f;
		tweaker.font.glyph_dimensions = {12.0f, 24.0f};
		tweaker.font.bearing_left = 2.0f;
		tweaker.font.bearing_right = 2.0f;
		tweaker.font.leading = leading;
		tweaker.font.tracking = 0.0f;
		tweaker.scroll_panel.bottom_left = {-300.0f, -200.0f};
		tweaker.scroll_panel.dimensions = {300.0f, 400.0f};
		tweaker.scroll_panel.padding = {8.0f, 8.0f};
		tweaker.scroll_panel.lines_count = tweaker_map_entries_count;
		tweaker.scroll_panel.line_height = leading;
	}

	// Setup the profile inspector.
	{
		float leading = 24.0f;
		profile_inspector.font.glyph_dimensions = {8.0f, 16.0f};
		profile_inspector.font.bearing_left = 2.0f;
		profile_inspector.font.bearing_right = 2.0f;
		profile_inspector.font.leading = leading;
		profile_inspector.font.tracking = 0.0f;
		profile_inspector.scroll_panel.bottom_left = {-370.0f, -120.0f};
		profile_inspector.scroll_panel.dimensions = {300.0f, 360.0f};
		profile_inspector.scroll_panel.padding = {8.0f, 8.0f};
		profile_inspector.scroll_panel.line_height = leading;
	}
}

static void game_destroy()
{
	for(int i = 0; i < world.colliders_count; ++i)
	{
		collider_destroy(&world.colliders[i]);
	}
}

static void game_send_message(audio::Message* message)
{
	enqueue_message(&message_queue, message);
}

static void process_messages_from_the_audio_thread()
{
	audio::Message message;
	while(dequeue_message(&message_queue, &message))
	{
		switch(message.code)
		{
			case audio::Message::Code::Note:
			{
				if(message.note.track == 0)
				{
					if(message.note.start)
					{
						dancer_position = {1.0f, 1.0f, 1.0f};
					}
					else
					{
						dancer_position = vector3_zero;
					}
				}
				break;
			}
			case audio::Message::Code::Oscilloscope_Settings:
			{
				render::oscilloscope.sample_rate = message.oscilloscope_settings.sample_rate;
				break;
			}
			case audio::Message::Code::Oscilloscope_Channel:
			{
				int channel_index = message.oscilloscope_channel.index;
				Oscilloscope::Channel* channel = &render::oscilloscope.channels[channel_index];
				channel->active = message.oscilloscope_channel.active;
				break;
			}
			case audio::Message::Code::Oscilloscope_Samples:
			{
				float* samples = message.oscilloscope_samples.array;
				int frames = message.oscilloscope_samples.frames;
				int channels = message.oscilloscope_samples.channels;
				int samples_count = frames * channels;
				for(int i = 0; i < channels; ++i)
				{
					oscilloscope_sample_data(&render::oscilloscope, i, samples, samples_count, i, channels);
				}
				DEALLOCATE(samples);
				break;
			}
			default:
			{
				// Any other message types are meant for the audio thread.
				ASSERT(false);
				break;
			}
		}
	}
}

static void main_update()
{
	process_messages_from_the_audio_thread();
	update_input_states();

	if(key_tapped(UserKey::Tilde))
	{
		tweaker_turn_on(&tweaker, !tweaker.on);
		profile::inspector_turn_on(&profile_inspector, false);
	}
	if(key_tapped(UserKey::F2))
	{
		profile::inspector_turn_on(&profile_inspector, !profile_inspector.on);
		tweaker_turn_on(&tweaker, false);
	}

	Vector3 velocity = vector3_zero;
	if(tweaker.on)
	{
		tweaker_update(&tweaker, &tweaker_map);
	}
	else if(profile_inspector.on)
	{
		profile::inspector_update(&profile_inspector);
	}
	else
	{
		// Update the player's movement state with the input.
		Vector2 d = vector2_zero;
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
		float l = length(d);
		if(l != 0.0f)
		{
			d.x /= l;
			d.y /= l;
			const float speed = 0.08f;
			velocity.x += speed * d.x;
			velocity.y += speed * d.y;
		}
		if(key_tapped(UserKey::Space))
		{
			audio::Message message;
			message.code = audio::Message::Code::Boop;
			message.boop.on = true;
			audio::system_send_message(&message);
		}
		if(key_released(UserKey::Space))
		{
			audio::Message message;
			message.code = audio::Message::Code::Boop;
			message.boop.on = false;
			audio::system_send_message(&message);
		}
	}

	Vector3 radius = {0.3f, 0.3f, 0.5f};
	position = collide_and_slide(position, radius, velocity, -0.096f * vector3_unit_z, &world);

	// Respawn if below the out-of-bounds plane.
	if(position.z < -4.0f)
	{
		position = vector3_zero;
	}

	render::system_update(position, dancer_position, &world, &tweaker, &profile_inspector);
}

// §5.2 Audio...................................................................

namespace audio {

namespace
{
	const int voice_count = 16;
	const int tracks_count = 5;
	const int streams_count = tracks_count + 1;
	const double first_section_length = 2.4;

	ConversionInfo conversion_info;
	DeviceDescription device_description;
	MessageQueue message_queue;
	float* mixed_samples;
	void* devicebound_samples;
	Voice voices[voice_count];
	VoiceEntry voice_map[voice_count];
	Stream streams[streams_count];
	Track tracks[tracks_count];
	Instrument instruments[tracks_count];
	Composer composer;
	History history;
	arandom::Sequence randomness;
	double time;
	bool boop_on;
}

TWEAKER_FLOAT_RANGE(master_volume, 0.0f, 0.0f, 1.0f);
TWEAKER_INT(boop_pitch, 69);

static void send_history_to_main_thread(History* history)
{
	for(int i = 0; i < history->count; ++i)
	{
		History::Event* event = &history->events[i];
		Message message;
		message.code = Message::Code::Note;
		message.note.start = event->started;
		message.note.track = event->track;
		game_send_message(&message);
	}
	history_erase(history);
}

static void send_oscilloscope_samples_to_main_thread(float* samples, int frames, int channels)
{
	// TODO: This allocates a buffer for every message and deallocates it on the
	// receiving side. It's … not the best.
	int samples_count = frames * channels;
	float* copy = ALLOCATE(float, samples_count);
	memcpy(copy, samples, sizeof(float) * samples_count);

	Message message;
	message.code = Message::Code::Oscilloscope_Samples;
	message.oscilloscope_samples.array = copy;
	message.oscilloscope_samples.frames = frames;
	message.oscilloscope_samples.channels = channels;
	game_send_message(&message);
}

static void process_messages_from_main_thread()
{
	Message message;
	while(dequeue_message(&message_queue, &message))
	{
		switch(message.code)
		{
			case Message::Code::Boop:
			{
				boop_on = message.boop.on;
				break;
			}
			default:
			{
				// Any other message types should be ones sent to the main
				// thread.
				ASSERT(false);
				break;
			}
		}
	}
}

void system_prepare_for_loop()
{
	u64 samples = device_description.channels * device_description.frames;

	// Setup mixing.
	mixed_samples = ALLOCATE(float, samples);
	devicebound_samples = ALLOCATE(u8, device_description.size);

	conversion_info.channels = device_description.channels;
	conversion_info.in.format = FORMAT_F32;
	conversion_info.in.stride = conversion_info.channels;
	conversion_info.out.format = device_description.format;
	conversion_info.out.stride = conversion_info.channels;

	float delta_time_per_frame = 1.0f / device_description.sample_rate;

	// Setup voices.

	for(int i = 0; i < voice_count; ++i)
	{
		Voice* voice = &voices[i];
		envelope_setup(&voice->amp_envelope);
		envelope_setup(&voice->pitch_envelope);
		arandom::seed_by_time(&voice->randomness);
	}

	voice_map_setup(voice_map, voice_count);

	// Setup the composer and tracks.

	arandom::seed_by_time(&randomness);

	compose_form(&composer);

	track_setup(&tracks[0]);
	track_generate(&tracks[0], first_section_length, &composer, &randomness);

	track_setup(&tracks[1]);
	tracks[1].octave = 4;
	tracks[1].octave_range = 0;
	tracks[1].style = 1;
	track_generate(&tracks[1], first_section_length, &composer, &randomness);

	track_setup(&tracks[2]);
	tracks[2].octave = 4;
	tracks[2].octave_range = 0;
	tracks[2].style = 1;
	track_generate(&tracks[2], first_section_length, &composer, &randomness);

	track_setup(&tracks[3]);
	tracks[3].octave = 4;
	tracks[3].octave_range = 1;
	tracks[3].style = 2;
	track_generate(&tracks[3], first_section_length, &composer, &randomness);

	track_setup(&tracks[4]);
	tracks[4].octave = 5;
	tracks[4].octave_range = 0;
	tracks[4].style = 3;
	track_generate(&tracks[4], first_section_length, &composer, &randomness);

	// Setup test instruments and streams.

	for(int i = 0; i < streams_count; ++i)
	{
		stream_create(&streams[i], device_description.frames, 1);
	}

	history_erase(&history);

	Instrument* kick = &instruments[0];
	kick->oscillator = Oscillator::Sine;
	kick->envelope_settings.amp.attack = 0.002f * device_description.sample_rate;
	kick->envelope_settings.amp.decay = 0.56f * device_description.sample_rate;
	kick->envelope_settings.amp.sustain = 0.0f;
	kick->envelope_settings.amp.release = 2.0f * device_description.sample_rate;
	kick->envelope_settings.pitch.use = true;
	kick->envelope_settings.pitch.semitones = 36;
	kick->envelope_settings.pitch.attack = 0.01f * device_description.sample_rate;
	kick->envelope_settings.pitch.decay = 0.06f * device_description.sample_rate;
	kick->envelope_settings.pitch.sustain = 0.0f;
	kick->envelope_settings.pitch.release = 0.0f * device_description.sample_rate;

	Effect* effect = instrument_add_effect(kick, EffectType::Low_Pass_Filter);
	LPF* lowpass = &effect->lowpass;
	lpf_set_corner_frequency(lowpass, 440.0f, delta_time_per_frame);

	effect = instrument_add_effect(kick, EffectType::Distortion);
	Distortion* distortion = &effect->distortion;
	distortion->gain = 5.0f;
	distortion->mix = 0.8f;

#if 0
	Instrument* open_hat = &instruments[1];
	open_hat->oscillator = Oscillator::Noise;
	open_hat->envelope_settings.attack = 0.002f * device_description.sample_rate;
	open_hat->envelope_settings.decay = 0.56f * device_description.sample_rate;
	open_hat->envelope_settings.sustain = 0.0f;
	open_hat->envelope_settings.release = 2.0f * device_description.sample_rate;
	open_hat->envelope_settings.pitch_envelope.use = false;
	open_hat->noise.passband = 48;
#endif

	Instrument* snare0 = &instruments[1];
	snare0->oscillator = Oscillator::Pulse;
	snare0->envelope_settings.amp.attack = 0.002f * device_description.sample_rate;
	snare0->envelope_settings.amp.decay = 0.2f * device_description.sample_rate;
	snare0->envelope_settings.amp.sustain = 0.1f;
	snare0->envelope_settings.amp.release = 0.6f * device_description.sample_rate;
	snare0->envelope_settings.pitch.use = true;
	snare0->envelope_settings.pitch.semitones = 36;
	snare0->envelope_settings.pitch.attack = 0.001f * device_description.sample_rate;
	snare0->envelope_settings.pitch.decay = 0.05f * device_description.sample_rate;
	snare0->envelope_settings.pitch.sustain = 0.0f;
	snare0->envelope_settings.pitch.release = 0.0f * device_description.sample_rate;
	snare0->pulse.width = 0.3f;

	Instrument* snare1 = &instruments[2];
	snare1->oscillator = Oscillator::Noise;
	snare1->envelope_settings.amp.attack = 0.05f * device_description.sample_rate;
	snare1->envelope_settings.amp.decay = 0.001f * device_description.sample_rate;
	snare1->envelope_settings.amp.sustain = 0.8f;
	snare1->envelope_settings.amp.release = 0.001f * device_description.sample_rate;
	snare1->envelope_settings.pitch.use = false;
	snare1->noise.passband = 24;

	effect = instrument_add_effect(snare1, EffectType::Overdrive);

	Instrument* lead = &instruments[3];
	lead->oscillator = Oscillator::FM_Sine;
	lead->fm.ratio = 0.333f;
	lead->fm.gain = 2400.0f;
	lead->envelope_settings.amp.attack = 0.2f * device_description.sample_rate;
	lead->envelope_settings.amp.decay = 0.1f * device_description.sample_rate;
	lead->envelope_settings.amp.sustain = 0.75f;
	lead->envelope_settings.amp.release = 0.0f * device_description.sample_rate;
	lead->envelope_settings.pitch.use = false;
	lead->envelope_settings.pitch.semitones = -12;
	lead->envelope_settings.pitch.attack = 0.1f * device_description.sample_rate;
	lead->envelope_settings.pitch.decay = 0.1f * device_description.sample_rate;
	lead->envelope_settings.pitch.sustain = 0.0f;
	lead->envelope_settings.pitch.release = 0.0f * device_description.sample_rate;

	effect = instrument_add_effect(lead, EffectType::Resonator);
	Resonator* resonator = &effect->resonator;
	resonator->mix = 0.4f;
	resonator->gain = 12.0f;
	sar_set_passband(&resonator->bank[0], pitch_to_frequency(66), device_description.sample_rate, 24.0f);
	sar_set_passband(&resonator->bank[1], pitch_to_frequency(73), device_description.sample_rate, 63.0f);
	sar_set_passband(&resonator->bank[2], pitch_to_frequency(55), device_description.sample_rate, 80.0f);

	Instrument* rim = &instruments[4];
	rim->oscillator = Oscillator::Noise;
	rim->envelope_settings.amp.attack = 0.008f * device_description.sample_rate;
	rim->envelope_settings.amp.decay = 0.14f * device_description.sample_rate;
	rim->envelope_settings.amp.sustain = 0.0f;
	rim->envelope_settings.amp.release = 0.0f * device_description.sample_rate;
	rim->envelope_settings.pitch.use = false;
	rim->noise.passband = 36;

	effect = instrument_add_effect(rim, EffectType::Resonator);
	resonator = &effect->resonator;
	resonator->mix = 0.9f;
	resonator->gain = 23.0f;
	sar_set_passband(&resonator->bank[0], pitch_to_frequency(89), device_description.sample_rate, 29.0f);
	sar_set_passband(&resonator->bank[1], pitch_to_frequency(85), device_description.sample_rate, 33.0f);
	sar_set_passband(&resonator->bank[2], pitch_to_frequency(88), device_description.sample_rate, 20.0f);

	effect = instrument_add_effect(rim, EffectType::Delay);
	Delay* delay = &effect->delay;
	delay->delay = 1.0f * device_description.sample_rate;
	delay->feedback = 0.2f;
	delay->mix = 0.5f;

	streams[1].pan = 0.4f;

	streams[2].pan = 0.4f;

	streams[3].pan = -0.1f;

	streams[4].volume = 0.6f;
	streams[4].pan = -0.7f;

	// Send some preparatory messages to the main thread.
	{
		Message message;

		message.code = Message::Code::Oscilloscope_Channel;
		message.oscilloscope_channel.index = 0;
		message.oscilloscope_channel.active = true;
		game_send_message(&message);

		message.code = Message::Code::Oscilloscope_Channel;
		message.oscilloscope_channel.index = 1;
		message.oscilloscope_channel.active = true;
		game_send_message(&message);

		message.code = Message::Code::Oscilloscope_Settings;
		message.oscilloscope_settings.sample_rate = device_description.sample_rate;
		game_send_message(&message);
	}
}

static void system_cleanup_after_loop()
{
	SAFE_DEALLOCATE(mixed_samples);
	SAFE_DEALLOCATE(devicebound_samples);
	for(int i = 0; i < streams_count; ++i)
	{
		stream_destroy(&streams[i]);
	}
	for(int i = 0; i < tracks_count; ++i)
	{
		instrument_destroy(&instruments[i]);
	}
}

static void system_update_loop()
{
	send_history_to_main_thread(&history);
	process_messages_from_main_thread();

	PROFILE_BEGIN_NAMED("generate_samples");

	int frames = device_description.frames;
	int sample_rate = device_description.sample_rate;
	double delta_time = device_description.frames / static_cast<double>(device_description.sample_rate);

	double section_finish_time = first_section_length;
	bool cue_time_reset = false;

	for(int i = 0; i < tracks_count; ++i)
	{
		Stream* stream = &streams[i];
		Track* track = &tracks[i];
		Instrument* instrument = &instruments[i];
		fill_with_silence(stream->samples, 0, stream->samples_count);
		if(time + delta_time < section_finish_time)
		{
			generate_oscillation(stream, 0, frames, track, i, instrument, voices, voice_map, voice_count, sample_rate, time, &history);
		}
		else
		{
			int regen_frame = (section_finish_time - time) * device_description.sample_rate;
			generate_oscillation(stream, 0, regen_frame, track, i, instrument, voices, voice_map, voice_count, sample_rate, time, &history);

			ASSERT(should_regenerate(track));
			transfer_unfinished_notes(track, section_finish_time);
			free_associated_voices(voice_map, voice_count, i);
			if(!cue_time_reset)
			{
				composer_update_state(&composer);
			}
			track_generate(track, section_finish_time, &composer, &randomness);

			cue_time_reset = true;

			generate_oscillation(stream, regen_frame, frames, track, i, instrument, voices, voice_map, voice_count, sample_rate, 0.0, &history);
		}
		apply_effects(instrument->effects, instrument->effects_count, stream);
	}

	Stream* stream = &streams[5];
	{
		stream->volume = 0.0f;
		if(boop_on)
		{
			stream->volume = 1.0f;
		}
		const float theta = pitch_to_frequency(boop_pitch);
		for(int i = 0; i < frames; ++i)
		{
			float t = static_cast<float>(i) / sample_rate + time;
			float value = sin(tau * theta * t);
			for(int j = 0; j < stream->channels; ++j)
			{
				stream->samples[stream->channels * i + j] = value;
			}
		}
	}

	PROFILE_END();

	// Combine the generated audio to a single compact array of samples.
	PROFILE_BEGIN_NAMED("mix_and_format");

	mix_streams(streams, streams_count, mixed_samples, frames, device_description.channels, master_volume);
	format_buffer_from_float(mixed_samples, devicebound_samples, device_description.frames, &conversion_info);

	PROFILE_END();

	send_oscilloscope_samples_to_main_thread(mixed_samples, frames, device_description.channels);

	if(cue_time_reset)
	{
		time = (time + delta_time) - section_finish_time;
	}
	else
	{
		time += delta_time;
	}
}

} // namespace audio

// 6. Profile...................................................................

namespace profile {

// §6.1 Spin Lock...............................................................

void spin_lock_acquire(SpinLock* lock)
{
	while(!atomic_compare_exchange(lock, 0, 1))
	{
		yield();
	}
}

void spin_lock_release(SpinLock* lock)
{
	while(!atomic_compare_exchange(lock, 1, 0))
	{
		yield();
	}
}

// §6.2 Caller..................................................................

struct Caller
{
	const char* name;

	Caller* parent;
	Caller** buckets;
	u32 bucket_count;
	u32 child_count;

	// used by the root caller of each thread to distinguish if that call tree
	// is active
	bool active;

	u64 started;
	u64 ticks;
	int calls;
	bool paused;
};

static void lock_this_thread();
static void unlock_this_thread();

static u32 hash_pointer(const char* name, u32 bucket_count)
{
	return (reinterpret_cast<size_t>(name) >> 5) & (bucket_count - 1);
}

static Caller** find_empty_child_slot(Caller** buckets, u32 bucket_count, const char* name)
{
	u32 index = hash_pointer(name, bucket_count);
	ASSERT(can_use_bitwise_and_to_cycle(bucket_count));
	u32 mask = bucket_count - 1;
	Caller** slot;
	for(slot = &buckets[index]; *slot; slot = &buckets[index & mask])
	{
		index += 1;
	}
	return slot;
}

static void resize(Caller* parent, u32 new_size)
{
	if(new_size < parent->bucket_count)
	{
		new_size = 2 * parent->bucket_count;
	}
	else
	{
		new_size = next_power_of_two(new_size - 1);
	}
	Caller** new_buckets = ALLOCATE(Caller*, new_size);
	for(u32 i = 0; i < parent->bucket_count; ++i)
	{
		if(parent->buckets[i])
		{
			Caller** slot = find_empty_child_slot(new_buckets, new_size, parent->buckets[i]->name);
			*slot = parent->buckets[i];
		}
	}
	DEALLOCATE(parent->buckets);
	parent->buckets = new_buckets;
	parent->bucket_count = new_size;
}

static void caller_create(Caller* caller, Caller* parent, const char* name)
{
	caller->name = name;
	caller->parent = parent;
	resize(caller, 2);
}

static void caller_destroy(Caller* caller)
{
	for(u32 i = 0; i < caller->bucket_count; ++i)
	{
		if(caller->buckets[i])
		{
			caller_destroy(caller->buckets[i]);
			DEALLOCATE(caller->buckets[i]);
		}
	}
	DEALLOCATE(caller->buckets);
}

static Caller* find_or_create(Caller* parent, const char* name)
{
	u32 index = hash_pointer(name, parent->bucket_count);
	ASSERT(can_use_bitwise_and_to_cycle(parent->bucket_count));
	u32 mask = parent->bucket_count - 1;
	for(Caller* caller = parent->buckets[index]; caller; caller = parent->buckets[index & mask])
	{
		if(caller->name == name)
		{
			return caller;
		}
		index += 1;
	}

	lock_this_thread();

	parent->child_count += 1;
	if(parent->child_count >= parent->bucket_count / 2)
	{
		resize(parent, parent->child_count);
	}

	Caller** slot = find_empty_child_slot(parent->buckets, parent->bucket_count, name);
	Caller* temp = ALLOCATE(Caller, 1);
	caller_create(temp, parent, name);
	*slot = temp;

	unlock_this_thread();

	return temp;
}

static void start_timing(Caller* caller)
{
	caller->calls += 1;
	caller->started = get_timestamp();
}

static void stop_timing(Caller* caller)
{
	caller->ticks += get_timestamp() - caller->started;
}

static void caller_reset(Caller* caller)
{
	caller->ticks = 0;
	caller->calls = 0;
	caller->started = get_timestamp();
	for(u32 i = 0; i < caller->bucket_count; ++i)
	{
		if(caller->buckets[i])
		{
			caller_reset(caller->buckets[i]);
		}
	}
}

static void caller_stop(Caller* caller)
{
	if(!caller->paused)
	{
		u64 t = get_timestamp();
		caller->ticks += t - caller->started;
		caller->started = t;
	}
}

static void caller_pause(Caller* caller, u64 pause_time)
{
	caller->ticks += pause_time - caller->started;
	caller->paused = true;
}

static void caller_unpause(Caller* caller, u64 unpause_time)
{
	caller->started = unpause_time;
	caller->paused = false;
}

#define COMPARE_TICKS(a, b)\
	a->ticks < b->ticks

DEFINE_INSERTION_SORT(Caller*, COMPARE_TICKS, ticks);

void caller_collect(Caller* caller, ThreadHistory* history, int thread, int indent)
{
	int index = history->indices[thread];
	Record* records = history->records[thread][index];
	int records_count = history->records_count[thread][index];
	int records_capacity = history->records_capacity[thread][index];
	ENSURE_ARRAY_SIZE(records, 1);
	history->records[thread][index] = records;
	history->records_capacity[thread][index] = records_capacity;

	Record* record = &records[records_count];
	history->records_count[thread][index] += 1;
	record->name = caller->name;
	record->ticks = caller->ticks;
	record->calls = caller->calls;
	record->indent = indent;

	// Form an array from the children hash table.
	const int children_cap = 8;
	Caller* children[children_cap];
	int children_count = 0;
	int bucket_count = caller->bucket_count;
	for(int i = 0; i < bucket_count; ++i)
	{
		Caller* child = caller->buckets[i];
		if(child && child->ticks > 0)
		{
			children[children_count] = child;
			children_count += 1;
			ASSERT(children_count < children_cap);
			if(children_count >= children_cap)
			{
				break;
			}
		}
	}

	if(children_count > 0)
	{
		insertion_sort_ticks(children, children_count);
	}
	for(int i = 0; i < children_count; ++i)
	{
		caller_collect(children[i], history, thread, indent + 1);
	}
}

// §6.3 Global Profile..........................................................

struct ThreadState
{
	SpinLock thread_lock;
	bool require_thread_lock;
	Caller* active_caller;
};

struct Root
{
	Caller caller;
	ThreadState* thread_state;
};

static const int thread_roots_cap = 8;

struct GlobalThreadsList
{
	Root roots[thread_roots_cap];
	int roots_count;
	SpinLock lock;
};

// All the global state is kept here.
namespace
{
	thread_local ThreadState thread_state;
	thread_local Caller* root;
	GlobalThreadsList threads_list;
}

static void lock_this_thread()
{
	if(thread_state.require_thread_lock)
	{
		spin_lock_acquire(&thread_state.thread_lock);
	}
}

static void unlock_this_thread()
{
	if(thread_state.require_thread_lock)
	{
		spin_lock_release(&thread_state.thread_lock);
	}
}

static void acquire_global_lock()
{
	spin_lock_acquire(&threads_list.lock);
}

static void release_global_lock()
{
	spin_lock_release(&threads_list.lock);
}

static Caller* add_root(ThreadState* state)
{
	Root* out = &threads_list.roots[threads_list.roots_count];
	threads_list.roots_count += 1;
	ASSERT(threads_list.roots_count < thread_roots_cap);
	out->thread_state = state;
	return &out->caller;
}

void begin_period(const char* name)
{
	Caller* parent = thread_state.active_caller;
	if(!parent)
	{
		return;
	}
	Caller* active = find_or_create(parent, name);
	start_timing(active);
	thread_state.active_caller = active;
}

void end_period()
{
	Caller* active = thread_state.active_caller;
	if(!active)
	{
		return;
	}
	stop_timing(active);
	thread_state.active_caller = active->parent;
}

void pause_period()
{
	u64 pause_time = get_timestamp();
	for(Caller* it = thread_state.active_caller; it; it = it->parent)
	{
		caller_pause(it, pause_time);
	}
}

void unpause_period()
{
	u64 unpause_time = get_timestamp();
	for(Caller* it = thread_state.active_caller; it; it = it->parent)
	{
		caller_unpause(it, unpause_time);
	}
}

void enter_thread(const char* name)
{
	acquire_global_lock();
	Caller* temp = add_root(&thread_state);
	release_global_lock();

	caller_create(temp, nullptr, name);

	lock_this_thread();

	thread_state.active_caller = temp;
	start_timing(temp);
	temp->active = true;
	root = temp;

	unlock_this_thread();
}

void exit_thread()
{
	lock_this_thread();

	stop_timing(root);
	root->active = false;
	thread_state.active_caller = nullptr;

	unlock_this_thread();
}

void reset_thread()
{
#if defined(PROFILE_ENABLED)
	lock_this_thread();

	caller_reset(root);
	for(Caller* it = thread_state.active_caller; it; it = it->parent)
	{
		it->calls = 1;
	}

	unlock_this_thread();
#endif
}

void cleanup()
{
	for(int i = 0; i < threads_list.roots_count; ++i)
	{
		Root* root = &threads_list.roots[i];
		if(root->caller.active)
		{
			caller_destroy(&root->caller);
		}
	}
}

void inspector_collect(Inspector* inspector, int thread)
{
#if defined(PROFILE_ENABLED)
	ThreadHistory* history = &inspector->history;

	lock_this_thread();
	spin_lock_acquire(&history->lock);

	if(!inspector->halt_collection)
	{
		int index = history->indices[thread];

		history->records_count[thread][index] = 0;
		caller_collect(root, history, thread, 0);

		history->indices[thread] = (index + 1) % thread_history_book_count;
	}

	spin_lock_release(&history->lock);
	unlock_this_thread();
#endif
}

} // namespace profile

// 7. OpenGL Function Loading...................................................

#if defined(OS_LINUX)
// glx.h includes X.h which has a typedef called Font. Defining _XTYPEDEF_FONT
// causes this typedef to not be included.
#define _XTYPEDEF_FONT
#include <GL/glx.h>
#if defined(Complex)
#undef Complex
#endif

#define GET_PROC(name) \
	(*glXGetProcAddress)(reinterpret_cast<const GLubyte*>(name))

#elif defined(OS_WINDOWS)
#define WIN32_LEAN_AND_MEAN 1
#define NOMINMAX
#include <Windows.h>

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
	p_glBlendFunc = reinterpret_cast<void (APIENTRYA*)(GLenum sfactor, GLenum dfactor)>(GET_PROC("glBlendFunc"));
	p_glClear = reinterpret_cast<void (APIENTRYA*)(GLbitfield)>(GET_PROC("glClear"));
	p_glDepthMask = reinterpret_cast<void (APIENTRYA*)(GLboolean)>(GET_PROC("glDepthMask"));
	p_glDisable = reinterpret_cast<void (APIENTRYA*)(GLenum)>(GET_PROC("glDisable"));
	p_glEnable = reinterpret_cast<void (APIENTRYA*)(GLenum)>(GET_PROC("glEnable"));
	p_glTexImage2D = reinterpret_cast<void (APIENTRYA*)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*)>(GET_PROC("glTexImage2D"));
	p_glViewport = reinterpret_cast<void (APIENTRYA*)(GLint, GLint, GLsizei, GLsizei)>(GET_PROC("glViewport"));

	p_glBindTexture = reinterpret_cast<void (APIENTRYA*)(GLenum, GLuint)>(GET_PROC("glBindTexture"));
	p_glDeleteTextures = reinterpret_cast<void (APIENTRYA*)(GLsizei, const GLuint*)>(GET_PROC("glDeleteTextures"));
	p_glDrawArrays = reinterpret_cast<void (APIENTRYA*)(GLenum, GLint, GLsizei)>(GET_PROC("glDrawArrays"));
	p_glDrawElements = reinterpret_cast<void (APIENTRYA*)(GLenum, GLsizei, GLenum, const void*)>(GET_PROC("glDrawElements"));
	p_glGenTextures = reinterpret_cast<void (APIENTRYA*)(GLsizei, GLuint*)>(GET_PROC("glGenTextures"));
	p_glTexSubImage2D = reinterpret_cast<void (APIENTRYA*)(GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void*)>(GET_PROC("glTexSubImage2D"));

	p_glActiveTexture = reinterpret_cast<void (APIENTRYA*)(GLenum)>(GET_PROC("glActiveTexture"));

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
	p_glUniform1f = reinterpret_cast<void (APIENTRYA*)(GLint, GLfloat)>(GET_PROC("glUniform1f"));
	p_glUniform1i = reinterpret_cast<void (APIENTRYA*)(GLint, GLint)>(GET_PROC("glUniform1i"));
	p_glUniform3fv = reinterpret_cast<void (APIENTRYA*)(GLint, GLsizei, const GLfloat*)>(GET_PROC("glUniform3fv"));
	p_glUniformMatrix4fv = reinterpret_cast<void (APIENTRYA*)(GLint, GLsizei, GLboolean, const GLfloat*)>(GET_PROC("glUniformMatrix4fv"));
	p_glUseProgram = reinterpret_cast<void (APIENTRYA*)(GLuint)>(GET_PROC("glUseProgram"));
	p_glVertexAttribPointer = reinterpret_cast<void (APIENTRYA*)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*)>(GET_PROC("glVertexAttribPointer"));

	p_glBindSampler = reinterpret_cast<void (APIENTRYA*)(GLuint, GLuint)>(GET_PROC("glBindSampler"));
	p_glDeleteSamplers = reinterpret_cast<void (APIENTRYA*)(GLsizei, const GLuint*)>(GET_PROC("glDeleteSamplers"));
	p_glGenSamplers = reinterpret_cast<void (APIENTRYA*)(GLsizei, GLuint*)>(GET_PROC("glGenSamplers"));
	p_glSamplerParameteri = reinterpret_cast<void (APIENTRYA*)(GLuint, GLenum, GLint)>(GET_PROC("glSamplerParameteri"));

	int failure_count = 0;

	failure_count += p_glBlendFunc == nullptr;
	failure_count += p_glClear == nullptr;
	failure_count += p_glDepthMask == nullptr;
	failure_count += p_glDisable == nullptr;
	failure_count += p_glEnable == nullptr;
	failure_count += p_glTexImage2D == nullptr;
	failure_count += p_glViewport == nullptr;

	failure_count += p_glBindTexture == nullptr;
	failure_count += p_glDeleteTextures == nullptr;
	failure_count += p_glDrawArrays == nullptr;
	failure_count += p_glDrawElements == nullptr;
	failure_count += p_glGenTextures == nullptr;
	failure_count += p_glTexSubImage2D == nullptr;

	failure_count += p_glActiveTexture == nullptr;

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
	failure_count += p_glUniform1f == nullptr;
	failure_count += p_glUniform1i == nullptr;
	failure_count += p_glUniform3fv == nullptr;
	failure_count += p_glUniformMatrix4fv == nullptr;
	failure_count += p_glUseProgram == nullptr;
	failure_count += p_glVertexAttribPointer == nullptr;

	failure_count += p_glBindSampler == nullptr;
	failure_count += p_glDeleteSamplers == nullptr;
	failure_count += p_glGenSamplers == nullptr;
	failure_count += p_glSamplerParameteri == nullptr;

	return failure_count == 0;
}

// 8. Compiler-Specific Implementations=========================================

// §8.1 Atomic Functions........................................................

#if defined(COMPILER_MSVC)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <intrin.h>
#endif

#if defined(COMPILER_MSVC)

bool atomic_flag_test_and_set(AtomicFlag* flag)
{
	return _InterlockedExchange(const_cast<volatile long*>(flag), 1L);
}

void atomic_flag_clear(AtomicFlag* flag)
{
	_InterlockedExchange(const_cast<volatile long*>(flag), 0L);
}

void atomic_int_store(AtomicInt* i, long value)
{
	_InterlockedExchange(const_cast<volatile long*>(i), value);
}

long atomic_int_load(AtomicInt* i)
{
	return _InterlockedOr(const_cast<volatile long*>(i), 0L);
}

long atomic_int_add(AtomicInt* augend, long addend)
{
	return InterlockedAdd(const_cast<volatile long*>(augend), addend);
}

long atomic_int_subtract(AtomicInt* minuend, long subtrahend)
{
	return InterlockedAdd(const_cast<volatile long*>(minuend), -subtrahend);
}

bool atomic_compare_exchange(volatile u32* p, u32 expected, u32 desired)
{
	return _InterlockedCompareExchange(p, desired, expected) == desired;
}

void yield()
{
#if defined(INSTRUCTION_SET_X86) || defined(INSTRUCTION_SET_X64)
	_mm_pause();
#elif defined(INSTRUCTION_SET_ARM)
	__yield();
#endif
}

u64 get_timestamp()
{
#if defined(INSTRUCTION_SET_X86) || defined(INSTRUCTION_SET_X64)
	return __rdtsc();
#elif defined(INSTRUCTION_SET_ARM)
	// ARMv6 has no performance counter and ARMv7-A and ARMv8-A can only
	// access their "Performance Monitor Unit" if the kernel enables
	// user-space to access it. So, it's too inconvenient to get at;
	// Instead, just fall back to the system call.
	return get_timestamp_from_system();
#endif
}

#elif defined(COMPILER_GCC)

bool atomic_flag_test_and_set(AtomicFlag* flag)
{
	return __atomic_test_and_set(flag, __ATOMIC_SEQ_CST);
}

void atomic_flag_clear(AtomicFlag* flag)
{
	__atomic_clear(flag, __ATOMIC_SEQ_CST);
}

void atomic_int_store(AtomicInt* i, long value)
{
	__atomic_store_n(const_cast<volatile long*>(i), value, __ATOMIC_SEQ_CST);
}

long atomic_int_load(AtomicInt* i)
{
	return __atomic_load_n(const_cast<volatile long*>(i), __ATOMIC_SEQ_CST);
}

long atomic_int_add(AtomicInt* augend, long addend)
{
	return __atomic_add_fetch(const_cast<volatile long*>(augend), addend, __ATOMIC_SEQ_CST);
}

long atomic_int_subtract(AtomicInt* minuend, long subtrahend)
{
	return __atomic_sub_fetch(const_cast<volatile long*>(minuend), subtrahend, __ATOMIC_SEQ_CST);
}

bool atomic_compare_exchange(volatile u32* p, u32 expected, u32 desired)
{
	u32 old = expected;
	return __atomic_compare_exchange_n(p, &old, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

// §8.2 Timing..................................................................

void yield()
{
#if defined(INSTRUCTION_SET_X86) || defined(INSTRUCTION_SET_X64)
	asm volatile ("pause");
#elif defined(INSTRUCTION_SET_ARM)
	asm volatile ("yield");
#endif
}

u64 get_timestamp()
{
#if defined(INSTRUCTION_SET_X86)
	u64 x;
	asm volatile ("rdtsc" : "=A" (x));
	return x;
#elif defined(INSTRUCTION_SET_X64)
	u64 a, d;
	asm volatile ("rdtsc" : "=a" (a), "=d" (d));
	return (d << 32) | a;
#elif defined(INSTRUCTION_SET_ARM)
	// ARMv6 has no performance counter and ARMv7-A and ARMv8-A can only
	// access their "Performance Monitor Unit" if the kernel enables
	// user-space to access it. So, it's too inconvenient to get at;
	// Instead, just fall back to the system call.
	return get_timestamp_from_system();
#endif
}

#endif // defined(COMPILER_GCC)

// 9. Platform-Specific Implementations=========================================

#if defined(OS_LINUX)

// §9.1 Logging Functions.......................................................

void log_add_message(LogLevel level, const char* format, ...)
{
	va_list arguments;
	va_start(arguments, format);
	FILE* stream;
	switch(level)
	{
		case LogLevel::Error:
		{
			stream = stderr;
			break;
		}
		case LogLevel::Debug:
		{
			stream = stdout;
			break;
		}
	}
	vfprintf(stream, format, arguments);
	va_end(arguments);
	fputc('\n', stream);
}

// §9.2 Clock Functions.........................................................

#include <ctime>

void initialise_clock(Clock* clock)
{
	struct timespec resolution;
	clock_getres(CLOCK_MONOTONIC, &resolution);
	s64 nanoseconds = resolution.tv_nsec + resolution.tv_sec * 1000000000;
	clock->frequency = static_cast<double>(nanoseconds) / 1.0e9;
}

double get_time(Clock* clock)
{
	struct timespec timestamp;
	clock_gettime(CLOCK_MONOTONIC, &timestamp);
	s64 nanoseconds = timestamp.tv_nsec + timestamp.tv_sec * 1000000000;
	return static_cast<double>(nanoseconds) * clock->frequency;
}

void go_to_sleep(Clock* clock, double amount_to_sleep)
{
	struct timespec requested_time;
	requested_time.tv_sec = 0;
	requested_time.tv_nsec = static_cast<s64>(1.0e9 * amount_to_sleep);
	clock_nanosleep(CLOCK_MONOTONIC, 0, &requested_time, nullptr);
}

u64 get_timestamp_from_system()
{
	timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return now.tv_sec * 1000000000 + now.tv_nsec;
}

// §9.3 Audio Functions.........................................................

#include <alsa/asoundlib.h>

#include <pthread.h>

namespace audio {

// §9.3.1 Device................................................................

static const int test_format_count = 5;

static Format test_formats[test_format_count] =
{
	FORMAT_F64,
	FORMAT_F32,
	FORMAT_S32,
	FORMAT_S16,
	FORMAT_S8,
};

snd_pcm_format_t get_equivalent_format(Format format)
{
	switch(format)
	{
		case FORMAT_U8:  return SND_PCM_FORMAT_U8;
		case FORMAT_S8:  return SND_PCM_FORMAT_S8;
		case FORMAT_U16: return SND_PCM_FORMAT_U16;
		case FORMAT_S16: return SND_PCM_FORMAT_S16;
		case FORMAT_U24: return SND_PCM_FORMAT_U24;
		case FORMAT_S24: return SND_PCM_FORMAT_S24;
		case FORMAT_U32: return SND_PCM_FORMAT_U32;
		case FORMAT_S32: return SND_PCM_FORMAT_S32;
		case FORMAT_F32: return SND_PCM_FORMAT_FLOAT;
		case FORMAT_F64: return SND_PCM_FORMAT_FLOAT64;
	}
	return SND_PCM_FORMAT_UNKNOWN;
}

static int finalize_hw_params(snd_pcm_t* pcm_handle, snd_pcm_hw_params_t* hw_params, bool override, u64* frames)
{
	int status;

	status = snd_pcm_hw_params(pcm_handle, hw_params);
	if(status < 0)
	{
		return -1;
	}

	snd_pcm_uframes_t buffer_size;
	status = snd_pcm_hw_params_get_buffer_size(hw_params, &buffer_size);
	if(status < 0)
	{
		return -1;
	}
	if(!override && buffer_size != *frames * 2)
	{
		return -1;
	}
	*frames = buffer_size / 2;

	return 0;
}

static int set_period_size(snd_pcm_t* pcm_handle, snd_pcm_hw_params_t* hw_params, bool override, u64* frames)
{
	int status;

	snd_pcm_hw_params_t* hw_params_copy;
	snd_pcm_hw_params_alloca(&hw_params_copy);
	snd_pcm_hw_params_copy(hw_params_copy, hw_params);

	if(!override)
	{
		return -1;
	}

	snd_pcm_uframes_t nearest_frames = *frames;
	status = snd_pcm_hw_params_set_period_size_near(pcm_handle, hw_params_copy, &nearest_frames, nullptr);
	if(status < 0)
	{
		return -1;
	}

	unsigned int periods = 2;
	status = snd_pcm_hw_params_set_periods_near(pcm_handle, hw_params_copy, &periods, nullptr);
	if(status < 0)
	{
		return -1;
	}

	return finalize_hw_params(pcm_handle, hw_params_copy, override, frames);
}

static int set_buffer_size(snd_pcm_t* pcm_handle, snd_pcm_hw_params_t* hw_params, bool override, u64* frames)
{
	int status;

	snd_pcm_hw_params_t* hw_params_copy;
	snd_pcm_hw_params_alloca(&hw_params_copy);
	snd_pcm_hw_params_copy(hw_params_copy, hw_params);

	if(!override)
	{
		return -1;
	}

	snd_pcm_uframes_t nearest_frames;
	nearest_frames = *frames * 2;
	status = snd_pcm_hw_params_set_buffer_size_near(pcm_handle, hw_params_copy, &nearest_frames);
	if(status < 0)
	{
		return -1;
	}

	return finalize_hw_params(pcm_handle, hw_params_copy, override, frames);
}

static bool open_device(const char* name, DeviceDescription* description, snd_pcm_t** out_pcm_handle)
{
	int status;

	snd_pcm_t* pcm_handle;
	status = snd_pcm_open(&pcm_handle, name, SND_PCM_STREAM_PLAYBACK, SND_PCM_NONBLOCK);
	if(status < 0)
	{
		LOG_ERROR("Couldn't open audio device \"%s\". %s", name, snd_strerror(status));
		return false;
	}
	*out_pcm_handle = pcm_handle;

	snd_pcm_hw_params_t* hw_params;
	snd_pcm_hw_params_alloca(&hw_params);
	status = snd_pcm_hw_params_any(pcm_handle, hw_params);
	if(status < 0)
	{
		LOG_ERROR("Couldn't get the hardware configuration. %s", snd_strerror(status));
		return false;
	}

	status = snd_pcm_hw_params_set_access(pcm_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
	if(status < 0)
	{
		LOG_ERROR("Couldn't set the hardware to interleaved access. %s", snd_strerror(status));
		return false;
	}

	Format test_format;
	status = -1;
	for(int i = 0; status < 0 && i < test_format_count; ++i)
	{
		test_format = test_formats[i];
		status = 0;
		snd_pcm_format_t pcm_format = get_equivalent_format(test_format);
		if(pcm_format == SND_PCM_FORMAT_UNKNOWN)
		{
			status = -1;
		}
		else
		{
			status = snd_pcm_hw_params_set_format(pcm_handle, hw_params, pcm_format);
		}
	}
	if(status < 0)
	{
		LOG_ERROR("Failed to obtain a suitable hardware audio format.");
		return false;
	}
	description->format = test_format;

	unsigned int channels = description->channels;
	status = snd_pcm_hw_params_set_channels(pcm_handle, hw_params, channels);
	if(status < 0)
	{
		status = snd_pcm_hw_params_get_channels(hw_params, &channels);
		if(status < 0)
		{
			LOG_ERROR("Couldn't set the channel count. %s", snd_strerror(status));
			return false;
		}
		description->channels = channels;
	}

	unsigned int resample = 1;
	status = snd_pcm_hw_params_set_rate_resample(pcm_handle, hw_params, resample);
	if(status < 0)
	{
		LOG_ERROR("Failed to enable resampling. %s", snd_strerror(status));
		return false;
	}

	unsigned int rate = description->sample_rate;
	status = snd_pcm_hw_params_set_rate_near(pcm_handle, hw_params, &rate, nullptr);
	if(status < 0)
	{
		LOG_ERROR("Couldn't set the sample rate. %s", snd_strerror(status));
		return false;
	}
	if(rate != description->sample_rate)
	{
		LOG_ERROR("Couldn't obtain the desired sample rate for the device.");
		return false;
	}
	description->sample_rate = rate;

	if(set_period_size(pcm_handle, hw_params, false, &description->frames) < 0 && set_buffer_size(pcm_handle, hw_params, false, &description->frames) < 0)
	{
		if(set_period_size(pcm_handle, hw_params, true, &description->frames) < 0)
		{
			LOG_ERROR("Couldn't set the desired period size and buffer size.");
			return false;
		}
	}

	snd_pcm_sw_params_t* sw_params;
	snd_pcm_sw_params_alloca(&sw_params);
	status = snd_pcm_sw_params_current(pcm_handle, sw_params);
	if(status < 0)
	{
		LOG_ERROR("Couldn't obtain the software configuration. %s", snd_strerror(status));
		return false;
	}

	status = snd_pcm_sw_params_set_avail_min(pcm_handle, sw_params, description->frames);
	if(status < 0)
	{
		LOG_ERROR("Couldn't set the minimum available samples. %s", snd_strerror(status));
		return false;
	}
	status = snd_pcm_sw_params_set_start_threshold(pcm_handle, sw_params, 1);
	if(status < 0)
	{
		LOG_ERROR("Couldn't set the start threshold. %s", snd_strerror(status));
		return false;
	}
	status = snd_pcm_sw_params(pcm_handle, sw_params);
	if(status < 0)
	{
		LOG_ERROR("Couldn't set software audio parameters. %s", snd_strerror(status));
		return false;
	}

	fill_remaining_device_description(description);

	return true;
}

static void close_device(snd_pcm_t* pcm_handle)
{
	if(pcm_handle)
	{
		snd_pcm_drain(pcm_handle);
		snd_pcm_close(pcm_handle);
	}
}

// §9.3.2 System Functions......................................................

namespace
{
	snd_pcm_t* pcm_handle;
	pthread_t thread;
	AtomicFlag quit;
}

static void* run_mixer_thread(void* argument)
{
	static_cast<void>(argument);

	PROFILE_THREAD_ENTER();

	device_description.channels = 2;
	device_description.format = FORMAT_S16;
	device_description.sample_rate = 44100;
	device_description.frames = 1024;
	fill_remaining_device_description(&device_description);
	if(!open_device("default", &device_description, &pcm_handle))
	{
		LOG_ERROR("Failed to open audio device.");
		// @Incomplete: probably should exit?
	}

	system_prepare_for_loop();
	
	int frame_size = conversion_info.channels * format_byte_count(conversion_info.out.format);

	while(atomic_flag_test_and_set(&quit))
	{
		system_update_loop();

		// Pass the completed audio to the device.

		PROFILE_BEGIN_NAMED("sleep_audio");

		int stream_ready = snd_pcm_wait(pcm_handle, 150);
		if(!stream_ready)
		{
			LOG_ERROR("ALSA device waiting timed out!");
		}

		PROFILE_END();

		profile::inspector_collect(&profile_inspector, 1);
		profile::reset_thread();

		PROFILE_BEGIN_NAMED("pass_audio_to_device");

		u8* buffer = static_cast<u8*>(devicebound_samples);
		snd_pcm_uframes_t frames_left = device_description.frames;
		while(frames_left > 0)
		{
			int frames_written = snd_pcm_writei(pcm_handle, buffer, frames_left);
			if(frames_written < 0)
			{
				int status = frames_written;
				if(status == -EAGAIN)
				{
					continue;
				}
				status = snd_pcm_recover(pcm_handle, status, 0);
				if(status < 0)
				{
					break;
				}
				continue;
			}
			buffer += frames_written * frame_size;
			frames_left -= frames_written;
		}

		PROFILE_END();
	}

	close_device(pcm_handle);
	system_cleanup_after_loop();

	PROFILE_THREAD_EXIT();

	LOG_DEBUG("Audio thread shut down.");

	return nullptr;
}

bool system_startup()
{
	atomic_flag_test_and_set(&quit);
	int result = pthread_create(&thread, nullptr, run_mixer_thread, nullptr);
	return result == 0;
}

void system_shutdown()
{
	// Signal the mixer thread to quit and wait here for it to finish.
	atomic_flag_clear(&quit);
	pthread_join(thread, nullptr);
}

void system_send_message(Message* message)
{
	enqueue_message(&message_queue, message);
}

} // namespace audio

// §9.4 Platform Main Functions.................................................

#include <X11/X.h>
#include <X11/Xlib.h>

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
	PROFILE_THREAD_ENTER();

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

	bool started = audio::system_startup();
	if(!started)
	{
		LOG_ERROR("Audio system failed startup.");
		return false;
	}
	game_create();

	return true;
}

static void main_destroy()
{
	game_destroy();
	audio::system_shutdown();
	render::system_terminate(functions_loaded);
	PROFILE_THREAD_EXIT();
	profile::cleanup();

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

		PROFILE_BEGIN_NAMED("swap_buffers");

		glXSwapBuffers(display, window);

		PROFILE_END();

		profile::inspector_collect(&profile_inspector, 0);
		profile::reset_thread();

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
		int keysyms[key_count] =
		{
			XK_space,
			XK_Left,
			XK_Up,
			XK_Right,
			XK_Down,
			XK_Home,
			XK_End,
			XK_Page_Up,
			XK_Page_Down,
			XK_asciitilde,
			XK_Return,
			XK_S,
			XK_V,
			XK_F2,
		};
		for(int i = 0; i < key_count; ++i)
		{
			int code = XKeysymToKeycode(display, keysyms[i]);
			keys_pressed[i] = keys[code / 8] & (1 << (code % 8));
		}

		PROFILE_BEGIN_NAMED("sleep");

		// Sleep off any remaining time until the next frame.
		double frame_thusfar = get_time(&frame_clock) - frame_start_time;
		if(frame_thusfar < frame_frequency)
		{
			go_to_sleep(&frame_clock, frame_frequency - frame_thusfar);
		}

		PROFILE_END();
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

// §9.1 Logging Functions.......................................................

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

// §9.2 Clock Functions.........................................................

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

u64 get_timestamp_from_system()
{
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);
	return now.QuadPart;
}

// §9.3 Audio...................................................................

#include <mmdeviceapi.h>
#include <Ks.h>
#include <KsMedia.h>
#include <Audioclient.h>
#include <Avrt.h>

#define SAFE_RELEASE(thing)\
	if((thing)) {(thing)->Release(); (thing) = nullptr;}

namespace audio {

// §9.3.1 Device................................................................

namespace
{
	const IID IID_MMDeviceEnumerator = __uuidof(MMDeviceEnumerator);
	const IID IID_IMMDeviceEnumerator = __uuidof(IMMDeviceEnumerator);
	const IID IID_IAudioClient = __uuidof(IAudioClient);
	const IID IID_IAudioRenderClient = __uuidof(IAudioRenderClient);

	AtomicFlag quit;
	HANDLE thread;
	DWORD thread_id;
	IAudioClient* audio_client;
	IAudioRenderClient* render_client;
	HANDLE buffer_event;
	u32 buffer_frame_capacity;
	u32 min_render_frames;
	u32 min_latency;
}

static bool is_integer_format(WAVEFORMATEX* format)
{
	return
		format->wFormatTag == WAVE_FORMAT_PCM ||
		format->wFormatTag == WAVE_FORMAT_EXTENSIBLE &&
		reinterpret_cast<WAVEFORMATEXTENSIBLE*>(format)->SubFormat == KSDATAFORMAT_SUBTYPE_PCM;
}

static bool is_float_format(WAVEFORMATEX* format)
{
	return
		format->wFormatTag == WAVE_FORMAT_IEEE_FLOAT ||
		format->wFormatTag == WAVE_FORMAT_EXTENSIBLE &&
		reinterpret_cast<WAVEFORMATEXTENSIBLE*>(format)->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
}

static bool translate_format(WAVEFORMATEX* format, Format* result)
{
	if(is_integer_format(format))
	{
		switch(format->wBitsPerSample)
		{
			case 8:  *result = FORMAT_U8;  return true;
			case 16: *result = FORMAT_S16; return true;
			case 24: *result = FORMAT_S24; return true;
			case 32: *result = FORMAT_S32; return true;
		}
	}
	else if(is_float_format(format))
	{
		*result = FORMAT_F32;
		return true;
	}
	return false;
}

static bool request_mix_format(DeviceDescription* description, WAVEFORMATEX** mix_format)
{
	WORD requested_channels = description->channels;
	WORD bit_rate = sizeof(float) * 8;
	DWORD sample_rate = description->sample_rate;
	WORD bytes_per_frame = requested_channels * (bit_rate / 8);

	WAVEFORMATEXTENSIBLE desired_format = {};
	desired_format.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
	desired_format.Format.nChannels = requested_channels;
	desired_format.Format.nSamplesPerSec = sample_rate;
	desired_format.Format.nAvgBytesPerSec = sample_rate * bytes_per_frame;
	desired_format.Format.nBlockAlign = bytes_per_frame;
	desired_format.Format.wBitsPerSample = bit_rate;
	desired_format.Format.cbSize = sizeof(desired_format) - sizeof(WAVEFORMATEX);

	desired_format.dwChannelMask = KSAUDIO_SPEAKER_STEREO;
	desired_format.SubFormat = KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
	desired_format.Samples.wValidBitsPerSample = bit_rate;

	HRESULT result = S_OK;
	
	WAVEFORMATEX* closest_format = nullptr;
	result = audio_client->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, reinterpret_cast<WAVEFORMATEX*>(&desired_format), &closest_format);
	if(FAILED(result))
	{
		LOG_ERROR("Couldn't get a compatible mix format.");
		return false;
	}

	// If the desired format is not supported, it will allocate and return the closest match
	// but if it IS supported, we have to allocate it ourselves and copy over the data
	// (it has to be allocated on the heap some way or another because we are keeping a
	// global copy called mix_format)
	if(result == S_OK && closest_format == nullptr)
	{
		closest_format = static_cast<WAVEFORMATEX*>(CoTaskMemAlloc(sizeof(WAVEFORMATEXTENSIBLE)));
		CopyMemory(closest_format, &desired_format, sizeof(WAVEFORMATEXTENSIBLE));
	}

	// check to see if the closest match has a sample type we can handle, otherwise there's an error
	Format device_format;
	bool translated = translate_format(closest_format, &device_format);
	if(FAILED(result) || !translated)
	{
		LOG_ERROR("device mix format did not support any compatible sample types");
		return false;
	}
	*mix_format = closest_format;

	description->format = device_format;
	description->sample_rate = closest_format->nSamplesPerSec;
	description->channels = closest_format->nChannels;

	return true;
}

static bool create_clients(IMMDeviceEnumerator* device_enumerator, IMMDevice* device, WAVEFORMATEX** mix_format, DeviceDescription* description)
{
	const int reftimes_per_second = 1e7;

	HRESULT result = S_OK;

	result = device->Activate(IID_IAudioClient, CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&audio_client));
	if(FAILED(result))
	{
		LOG_ERROR("Failed to create an audio client.");
		return false;
	}

	bool request_success = request_mix_format(description, mix_format);
	if(!request_success)
	{
		return false;
	}

	DWORD stream_flags = AUDCLNT_STREAMFLAGS_NOPERSIST | AUDCLNT_STREAMFLAGS_EVENTCALLBACK;
	REFERENCE_TIME duration = reftimes_per_second * 2 * device_description.frames / static_cast<double>(device_description.sample_rate);
	REFERENCE_TIME periodicity = 0; // Must be 0 when using AUDCLNT_SHAREMODE_SHARED
	result = audio_client->Initialize(AUDCLNT_SHAREMODE_SHARED, stream_flags, duration, periodicity, *mix_format, nullptr);
	if(FAILED(result))
	{
		LOG_ERROR("Failed to initialize the audio client.");
		return false;
	}

	result = audio_client->GetService(IID_IAudioRenderClient, reinterpret_cast<void**>(&render_client));
	if(FAILED(result))
	{
		LOG_ERROR("Failed to get the render client from the audio client.");
		return false;
	}

	// Create an event handle and register it for buffer-event notifications.
	buffer_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	result = audio_client->SetEventHandle(buffer_event);
	if(FAILED(result))
	{
		LOG_ERROR("Failed to set the buffer event handle.");
		return false;
	}

	result = audio_client->GetBufferSize(&buffer_frame_capacity);
	if(FAILED(result))
	{
		LOG_ERROR("Couldn't get the buffer frames count.");
		return false;
	}
	ASSERT(description->frames <= buffer_frame_capacity);
	description->frames = MIN(description->frames, buffer_frame_capacity);

	REFERENCE_TIME device_period = 0;
	result = audio_client->GetDevicePeriod(&device_period, nullptr);
	if(FAILED(result))
	{
		LOG_ERROR("Couldn't retrieve the update scheduling period.");
		return false;
	}
	min_render_frames = (device_period * (*mix_format)->nSamplesPerSec + reftimes_per_second - 1) / reftimes_per_second;
	description->frames = MAX(description->frames, min_render_frames);

	REFERENCE_TIME latency = 0;
	result = audio_client->GetStreamLatency(&latency);
	if(FAILED(result))
	{
		LOG_ERROR("Couldn't determine the client stream latency.");
		return false;
	}
	min_latency = latency + device_period;

	fill_remaining_device_description(description);

	return true;
}

static bool start_device()
{
	HRESULT result = audio_client->Start();
	if(FAILED(result))
	{
		LOG_ERROR("The audio client failed to start playback.");
		return false;
	}
	return true;
}

static void stop_device()
{
	HRESULT result = audio_client->Stop();
	if(FAILED(result))
	{
		LOG_ERROR("The audio client failed to stop playback.");
	}
}

static void cleanup_after_opening(IMMDeviceEnumerator* device_enumerator, IMMDevice* device, WAVEFORMATEX* mix_format)
{
	CoTaskMemFree(mix_format);
	SAFE_RELEASE(device);
	SAFE_RELEASE(device_enumerator);

	CoUninitialize();
}

static bool open_device()
{
	IMMDeviceEnumerator* device_enumerator = nullptr;
	IMMDevice* device = nullptr;
	WAVEFORMATEX* mix_format = nullptr;

	HRESULT result = S_OK;

	result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
	if(FAILED(result))
	{
		LOG_ERROR("Failed to initialise the COM library for mmdevapi.");
		cleanup_after_opening(device_enumerator, device, mix_format);
		return false;
	}

	result = CoCreateInstance(IID_MMDeviceEnumerator, nullptr, CLSCTX_ALL, IID_IMMDeviceEnumerator, reinterpret_cast<LPVOID*>(&device_enumerator));
	if(FAILED(result))
	{
		LOG_ERROR("Failed to create a device enumerator.");
		cleanup_after_opening(device_enumerator, device, mix_format);
		return false;
	}

	result = device_enumerator->GetDefaultAudioEndpoint(EDataFlow::eRender, ERole::eConsole, &device);
	if(FAILED(result))
	{
		LOG_ERROR("Failed to retrieve the default render endppoint.");
		cleanup_after_opening(device_enumerator, device, mix_format);
		return false;
	}

	bool clients_created = create_clients(device_enumerator, device, &mix_format, &device_description);
	bool started = start_device();

	cleanup_after_opening(device_enumerator, device, mix_format);

	return clients_created && started;
}

static void close_device()
{
	if(buffer_event)
	{
		CloseHandle(buffer_event);
	}
	if(audio_client)
	{
		stop_device();
	}
	SAFE_RELEASE(audio_client);
	SAFE_RELEASE(render_client);
}

static const char* lookup_audclnt_error_message(HRESULT result)
{
#define M(code)\
	case code: return #code

	switch(result)
	{
		M(AUDCLNT_E_NOT_INITIALIZED);
		M(AUDCLNT_E_ALREADY_INITIALIZED);
		M(AUDCLNT_E_WRONG_ENDPOINT_TYPE);
		M(AUDCLNT_E_DEVICE_INVALIDATED);
		M(AUDCLNT_E_NOT_STOPPED);
		M(AUDCLNT_E_BUFFER_TOO_LARGE);
		M(AUDCLNT_E_OUT_OF_ORDER);
		M(AUDCLNT_E_UNSUPPORTED_FORMAT);
		M(AUDCLNT_E_INVALID_SIZE);
		M(AUDCLNT_E_DEVICE_IN_USE);
		M(AUDCLNT_E_BUFFER_OPERATION_PENDING);
		M(AUDCLNT_E_THREAD_NOT_REGISTERED);
		M(AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED);
		M(AUDCLNT_E_ENDPOINT_CREATE_FAILED);
		M(AUDCLNT_E_SERVICE_NOT_RUNNING);
		M(AUDCLNT_E_EVENTHANDLE_NOT_EXPECTED);
		M(AUDCLNT_E_EXCLUSIVE_MODE_ONLY);
		M(AUDCLNT_E_BUFDURATION_PERIOD_NOT_EQUAL);
		M(AUDCLNT_E_EVENTHANDLE_NOT_SET);
		M(AUDCLNT_E_INCORRECT_BUFFER_SIZE);
		M(AUDCLNT_E_BUFFER_SIZE_ERROR);
		M(AUDCLNT_E_CPUUSAGE_EXCEEDED);
		M(AUDCLNT_E_BUFFER_ERROR);
		M(AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED);
		M(AUDCLNT_E_INVALID_DEVICE_PERIOD);
		M(AUDCLNT_S_BUFFER_EMPTY);
		M(AUDCLNT_S_THREAD_ALREADY_REGISTERED);
		M(AUDCLNT_S_POSITION_STALLED);
#define AUDCLNT_S_NO_SINGLE_PROCESS AUDCLNT_SUCCESS(0x00d)
		M(AUDCLNT_S_NO_SINGLE_PROCESS);
		default: return "unknown error";
	}

#undef M
}

// §9.3.2 System Functions......................................................

static void pass_audio_to_device()
{
	HRESULT result = S_OK;

	UINT32 render_frame_count = device_description.frames;
	UINT32 frames_available = 0;

	// Loop and wait until enough of the buffer is freed to write the samples.
	// Note that this shouldn't loop at all unless the render client buffer is
	// full and thus has no room to write to.
	while(render_frame_count > frames_available)
	{
		DWORD wait_result = WaitForSingleObject(buffer_event, 1000);
		if(wait_result == WAIT_OBJECT_0)
		{
			UINT32 padding_frames;
			result = audio_client->GetCurrentPadding(&padding_frames);
			if(FAILED(result))
			{
				LOG_ERROR("Couldn't determine the padding for the audio render client buffer.");
			}
			frames_available = buffer_frame_capacity - padding_frames;
		}
		else if(FAILED(wait_result))
		{
			LOG_ERROR("The audio client callback didn't ever get signaled.");
		}
	}

	int frames_mixed = device_description.frames;

	BYTE* data = nullptr;
	result = render_client->GetBuffer(frames_mixed, &data);
	if(FAILED(result))
	{
		const char* message = lookup_audclnt_error_message(result);
		LOG_ERROR("Failed to get a packet to buffer the audio data to. Error Code: %s", message);
	}
	else
	{
		memcpy(data, devicebound_samples, device_description.size);

		DWORD flags;
		if(frames_mixed > 0)
		{
			flags = 0;
		}
		else
		{
			flags = AUDCLNT_BUFFERFLAGS_SILENT;
		}
		result = render_client->ReleaseBuffer(frames_mixed, flags);
		if(FAILED(result))
		{
			LOG_ERROR("Failed to release the audio buffer packet.");
		}
	}
}

DWORD WINAPI run_thread(_In_ LPVOID parameter)
{
	static_cast<void>(parameter);

	PROFILE_THREAD_ENTER();

	device_description.frames = 1024;
	device_description.format = FORMAT_F32;
	device_description.sample_rate = 44100;
	device_description.channels = 2;
	bool opened = open_device();
	if(!opened)
	{
		LOG_ERROR("Failed to start the audio device.");
	}

	system_prepare_for_loop();

	DWORD task_index = 0;
	HANDLE task = AvSetMmThreadCharacteristics(TEXT("Pro Audio"), &task_index);
	if(!task)
	{
		LOG_ERROR("Thread priority was not escalated as requested.");
	}

	while(atomic_flag_test_and_set(&quit))
	{
		system_update_loop();

		profile::inspector_collect(&profile_inspector, 1);
		profile::reset_thread();

		pass_audio_to_device();
	}

	if(task)
	{
		AvRevertMmThreadCharacteristics(task);
	}

	system_cleanup_after_loop();
	close_device();

	PROFILE_THREAD_EXIT();

	LOG_DEBUG("Audio thread shut down.");

	return 0;
}

bool system_startup()
{
	atomic_flag_test_and_set(&quit);
	thread = CreateThread(nullptr, 0, run_thread, nullptr, 0, &thread_id);
	return thread;
}

void system_shutdown()
{
	// Signal the thread to quit and wait here for it to finish.
	atomic_flag_clear(&quit);
	WaitForSingleObject(thread, INFINITE);
	CloseHandle(thread);
}

void system_send_message(Message* message)
{
	enqueue_message(&message_queue, message);
}

} // namespace audio

// §9.4 Platform Main Functions.................................................

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
	PROFILE_THREAD_ENTER();

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
    window = CreateWindowExA(WS_EX_APPWINDOW, MAKEINTATOM(registered_class), app_name, window_style, CW_USEDEFAULT, CW_USEDEFAULT, window_width, window_height, nullptr, nullptr, instance, nullptr);
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
        LOG_ERROR("Couldn't set this thread's rendering context (wglMakeCurrent failed).");
        return false;
    }

    arandom::seed(time(nullptr));

    ogl_functions_loaded = ogl_load_functions();
    if(!ogl_functions_loaded)
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

	bool started = audio::system_startup();
	if(!started)
	{
		LOG_ERROR("Audio system failed startup.");
		return false;
	}

	game_create();

    return true;
}

static void main_destroy()
{
	game_destroy();
	audio::system_shutdown();
    render::system_terminate(ogl_functions_loaded);
	PROFILE_THREAD_EXIT();
	profile::cleanup();

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

		main_update();
		SwapBuffers(device_context);

		profile::inspector_collect(&profile_inspector, 0);
		profile::reset_thread();

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
		const int virtual_keys[key_count] =
		{
			VK_SPACE,
			VK_LEFT,
			VK_UP,
			VK_RIGHT,
			VK_DOWN,
			VK_HOME,
			VK_END,
			VK_PRIOR,
			VK_NEXT,
			VK_OEM_3,
			VK_RETURN,
			'S',
			'V',
			VK_F2,
		};
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
