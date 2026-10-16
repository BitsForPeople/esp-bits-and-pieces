#pragma once
/*
 * Copyright 2026, <https://github.com/BitsForPeople>
 */
#include <cstdint>
#include <type_traits>
#include <array>


namespace simd {

    template<unsigned I>
    requires (I < 8)
    struct QReg;

    namespace {
        template<unsigned X>
        constexpr bool check_qr(const QReg<X>&) {
            return true;
        }

        template<unsigned X>
        constexpr unsigned qr_num(const QReg<X>&) {
            return X;
        }
    }


    template<typename T>
    concept QR_t = requires (const T& x) {{qr_num(x)};};

    template<unsigned I>
    requires (I < 8)
    struct QReg {
        static constexpr unsigned N = I;

        constexpr QReg(void) = default;
        template<unsigned J>
        inline const QReg& operator =(const QReg<J>&) const {
            if constexpr (J != N) {
                asm volatile (
                    "MV.QR q%[d], q%[s]" "\n"
                    :
                    : 
                        [s] "n" (J),
                        [d] "n" (N)
                );
            }
            return *this;
        }

        constexpr unsigned num(void) const {
            return N;
        }

        template<unsigned J>
        constexpr std::strong_ordering operator <=>(const QReg<J>&) const {
            return N <=> J;
        }

        template<QR_t Q1>
        static constexpr bool sameAs = (N == Q1::N);

        template<QR_t Q1, QR_t ... Qs>
        static constexpr bool anyOf = (sameAs<Q1> || (sameAs<Qs> || ...));

        template<QR_t Q1, QR_t ... Qs>
        static constexpr bool noneOf = !anyOf<Q1,Qs...>;
    };


    template<typename Q, typename Q1, typename ... Qs>
    concept qr_anyOf = QR_t<Q> && Q::template anyOf<Q1,Qs...>;

    template<typename Q, typename Q1>
    concept qr_sameAs = QR_t<Q> && Q::template sameAs<Q1>;

    template<typename Q, typename Q1, typename ... Qs>
    concept qr_noneOf = QR_t<Q> && QR_t<Q1> && (QR_t<Qs>&&...) && !qr_anyOf<Q,Q1,Qs...>;

    template<typename Q, unsigned X>
    concept qr_below = QR_t<Q> && Q::N < X;

    template<typename Q, unsigned X>
    concept qr_above = QR_t<Q> && Q::N > X;

    template<typename Q, unsigned L, unsigned H>
    concept qr_in_range = QR_t<Q> && (Q::N >= L) && (Q::N <= H);


    using Q0 = QReg<0>;
    using Q1 = QReg<1>;
    using Q2 = QReg<2>;
    using Q3 = QReg<3>;
    using Q4 = QReg<4>;
    using Q5 = QReg<5>;
    using Q6 = QReg<6>;
    using Q7 = QReg<7>;

    constexpr Q0 q0 {};
    constexpr Q1 q1 {};
    constexpr Q2 q2 {};
    constexpr Q3 q3 {};
    constexpr Q4 q4 {};
    constexpr Q5 q5 {};
    constexpr Q6 q6 {};
    constexpr Q7 q7 {};



    template<typename T>
    requires ((16 % sizeof(T)) == 0)
    struct Vec {
        using type = std::remove_cv_t<T>;

        static constexpr std::size_t ELEM_SZ = sizeof(type);
        static constexpr std::size_t SIZE = 16/ELEM_SZ;

        alignas(16) std::array<type, SIZE> data;

        constexpr Vec(void) {
            if(std::is_constant_evaluated()) {
                data = decltype(data) {};
            }
        }

        template<typename U>
        constexpr Vec(const Vec<U>& other) {
            this->operator =(other);
        }

        template<typename U>
        constexpr Vec& operator =(const Vec<U>& other) {
            if(std::is_constant_evaluated()) {
                this->data = other.data;
            } else {
                asm (
                    "LD.QR q0, %[src], 0" "\n"
                    "ST.QR q0, %[dst], 0" "\n"
                    : 
                        "=m" (this->data)
                    :
                        "m" (other.data),
                        [src] "r" (&other.data),
                        [dst] "r" (&this->data)
                );
            }
        }

        constexpr T& operator [](const unsigned i) {
            return this->data[i];
        }

        constexpr const T& operator [](const unsigned i) const {
            return this->data[i];
        }
        
        template<typename U>
        Vec<U>& as(void) {
            return reinterpret_cast<Vec<U>&>(*this);
        }

        template<typename U>
        const Vec<U>& as(void) const {
            return reinterpret_cast<const Vec<U>&>(*this);
        }

    };

    /**
     * @brief Utility routines for working with the PIE's Q registers
     * 
     */
    namespace qtil {

        namespace {

            template<typename T>
            static inline T& as(void* dst) {
                return *(T*)dst;
            }

            template<typename P>
            static inline void incp(P*& ptr, int32_t inc) {
                ptr = (P*)((uintptr_t)ptr + inc);
            }

            template<typename T, typename P>
            static inline void st(P*& ptr, uint32_t v) {
                as<T>(ptr) = v;
                incp(ptr,sizeof(T));
            }

            static inline constexpr bool known(const bool c) {
                return __builtin_constant_p(c) && c;
            }

            static inline constexpr bool maybe(const bool c) {
                return !known(!c);
            }

        } // anon namespace

        /**
         * @brief An empty struct which can be used to make dependencies between
         * inline assembly blocks visible to the compiler.
         * 
         * An assembly block which modifies some state invisible to the compiler declares
         * a dependency on a \c Dep instance as <tt>"+m" (dep)</tt>; an assembly block 
         * which depends on a previously modified state but doesn't modify it declares
         * <tt>"m" (dep)</tt>.
         * 
         * <pre>
         * qtil::Dep dep {};
         * asm ("EE.ZERO.Q q0": "+m" (dep)); // Modifies/sets up state (contents of q0)
         * 
         * uint32_t data;
         * asm ("EE.MOVI.32.A q0, %[data], 0" : [data] "=r" (data) : "m" (dep)); // Depends on the previous block
         * return data;
         * </pre>
         * 
         * This way, dependencies become visible to the compiler, and it can optimize accordingly.
         * If one assembly block is determined to be "dead" code, the blocks it depends on may also
         * become dead code and optimized away.
         * 
         * To track dependencies across function calls, a \c qtil::Dep& is passed as an argument.
         * 
         */
        struct Dep {

        };

        template<simd::QR_t D, simd::QR_t S>
        inline void mv(Dep& dep) {
            if constexpr (D::N != S::N) {
                asm (
                    "MV.QR q%[d], q%[s]"
                    :
                        "+m" (dep)
                    :
                        [d] "n" (D::N),
                        [s] "n" (S::N)
                );
            }
        }

        template<simd::QR_t QL, simd::QR_t QH, unsigned WIDTH = 8>
        requires (QL::N != QH::N && (WIDTH == 8 || WIDTH == 16 || WIDTH == 32))
        inline void swap(Dep& dep) {
            asm (
                "EE.VUNZIP.%[w] q%[ql], q%[qh]" "\n"
                "EE.VZIP.%[w] q%[qh], q%[ql]"
                :
                    "+m" (dep)
                :
                    [ql] "n" (QL::N),
                    [qh] "n" (QH::N),
                    [w] "n" (WIDTH)
            );
        }

        template<simd::QR_t QL, simd::QR_t QH>
        requires (QL::N != QH::N)
        inline void swap8(Dep& dep) {
            swap<QL,QH,8>(dep);
        }

        template<simd::QR_t Q, unsigned W>
        requires (W < 4)
        inline uint32_t getw(Dep& dep) {
            uint32_t data;
            asm (
                "EE.MOVI.32.A q%[qr], %[data], %[w]" "\n"
                :
                    [data] "=r" (data)
                :
                    [qr] "n" (Q::N),
                    "m" (dep),
                    [w] "n" (W)
            );
            return data;
        }


        template<typename T>
        concept ldable_t = (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4) && alignof(T) == sizeof(T);

        template<simd::QR_t Q, ldable_t T>
        inline void ldbc(const auto& src, Dep& dep) {
            asm (
                "EE.VLDBC.%[w] q%[qr], %[src]"
                :
                    "+m" (dep)
                :
                    [qr] "n" (Q::N),
                    [w] "n" (sizeof(T)*8),
                    [src] "r" (&src),
                    "m" (*(const T*)&src)
            );
        }

        template<simd::QR_t Q, ldable_t T, unsigned INC, typename S>
        requires ((INC % sizeof(T)) == 0)
        inline void ldbcip(S*& src, Dep& dep) {
            asm (
                "EE.VLDBC.%[w].IP q%[qr], %[src], %[inc]"
                :
                    "+m" (dep),
                    [src] "+r" (src)
                :
                    [qr] "n" (Q::N),
                    [w] "n" (sizeof(T)*8),
                    "m" (*(const T*)src),
                    [inc] "n" (INC)
            );
        }

        template<simd::QR_t Q, unsigned OFF = 0>
        requires ((OFF % 16) == 0)
        inline void stqr(void* const dst, Dep& dep) {
            asm (
                "ST.QR q%[qr], %[dst], %[off]"
                :
                    "=m" (*(uint8_t(*)[16])dst)
                :
                    "m" (dep),
                    [qr] "n" (Q::N),
                    [dst] "r" (dst),
                    [off] "n" (OFF)
            );
        }

        template<simd::QR_t Q>
        inline void zero(Dep& dep) {
            asm (
                "EE.ZERO.Q q%[qr]"
                :
                    "+m" (dep)
                :
                    [qr] "n" (Q::N)
            );
        }


        template<simd::QR_t Q>
        inline void rotrq(const uint32_t bytes, Dep& dep) {
            asm (
                "WUR.SAR_BYTE %[s]" "\n"
                "EE.SRC.Q q%[qr], q%[qr], q%[qr]"
                :
                    "+m" (dep)
                :
                    [s] "r" (bytes),
                    [qr] "n" (Q::N)
            );
        }


        template<simd::QR_t Q>
        inline void* storeQ(void* dst, Dep& dep) {
            if(known(((uintptr_t)dst % 16) == 0)) { // This check isn't worth doing at runtime.
                asm (
                    "EE.VST.128.IP q%[qr], %[dst], 16" "\n"
                    :
                        [dst] "+r" (dst),
                        "=m" (*(uint8_t(*)[16])dst)
                    :
                        [qr] "n" (Q::N),
                        "m" (dep)
                );
            } else
            if((uintptr_t)dst % 8 == 0) {
                asm (
                    "EE.VST.L.64.IP q%[qr], %[dst], 8" "\n"
                    "EE.VST.H.64.IP q%[qr], %[dst], 8" "\n"
                    :
                        [dst] "+r" (dst),
                        "=m" (*(uint8_t(*)[16])dst)
                    :
                        [qr] "n" (Q::N),
                        "m" (dep)
                );
            } else {
                uint32_t* const d32 = (uint32_t*)dst;
                d32[0] = getw<Q,0>(dep);
                d32[1] = getw<Q,1>(dep);
                d32[2] = getw<Q,2>(dep);
                d32[3] = getw<Q,3>(dep);
                dst = d32 + 4;
            }

            return dst;
        }

        template<simd::QR_t Q> // Q clobbered!
        // [[gnu::optimize("Os")]]
        inline void storeQ(void* dst, const uint32_t byteCnt, Dep& dep) {
            [[assume(byteCnt <= 15)]];

            if((byteCnt % 4) != 0) {
                uint32_t data = getw<Q,0>(dep);
                if(byteCnt & 2) {
                    st<uint16_t>(dst,data);
                    data = data >> 16;
                }
                if(byteCnt & 1) {
                    st<uint8_t>(dst,data);
                }
                rotrq<Q>(byteCnt % 4, dep);
            }

            const uint32_t words = byteCnt / sizeof(uint32_t);
            if(words != 0) {
                uint32_t* const d32 = (uint32_t*)dst;
                d32[0] = getw<Q,0>(dep);
                switch(words) {
                    case 3:
                        d32[2] = getw<Q,2>(dep);
                        [[fallthrough]];
                    case 2:
                        d32[1] = getw<Q,1>(dep);
                };
            }
        }



        template<simd::QR_t QL, simd::QR_t QH>
        requires (simd::qr_in_range<QL,0,5> && simd::qr_in_range<QH,0,5>) // q6+q7 used as temporaries, clobbered
        inline void* store2Q(void* dst, Dep& dep) {

            if(known(((uintptr_t)dst % 16) == 0)) { 
                asm (
                    "EE.VST.128.IP q%[ql], %[dst], 16" "\n"
                    "EE.VST.128.IP q%[qh], %[dst], 16" "\n"
                    : 
                        [dst] "+r" (dst),
                        "=m" (*(uint8_t(*)[32])dst)
                    : 
                        "m" (dep),
                        [ql] "n" (QL::N),
                        [qh] "n" (QH::N)
                );
            } else
            if(((uintptr_t)dst % 8) == 0) {
                asm (
                    "EE.VST.L.64.IP q%[ql], %[dst], 8" "\n"
                    "EE.VST.H.64.IP q%[ql], %[dst], 8" "\n"
                    "EE.VST.L.64.IP q%[qh], %[dst], 8" "\n"
                    "EE.VST.H.64.IP q%[qh], %[dst], 8" "\n"
                    :
                        [dst] "+r" (dst),
                        "=m" (*(uint8_t(*)[32])dst)
                    :
                        [ql] "n" (QL::N),
                        [qh] "n" (QH::N),
                        "m" (dep)
                );
            } else {
                const uint32_t off = 16- ((uintptr_t)dst % 16);
                asm (
                    "EE.LD.128.USAR.IP q6, %[dst], 32" "\n" // load the vector containing the first byte into q6
                    "EE.VLD.128.IP q7, %[dst], -32" "\n" // load the vector containing the last byte into q7

                    "EE.SRC.Q q6, q6, q6" "\n" // left-align the lower bytes we want to keep
                    "EE.SRC.Q q7, q7, q7" "\n" // right-align the upper bytes we want to keep

                    "WUR.SAR_BYTE %[off]" "\n"

                    "EE.SRCQ.128.ST.INCP q6, q%[ql], %[dst]" "\n"
                    "EE.SRCQ.128.ST.INCP q%[ql], q%[qh], %[dst]" "\n"
                    "EE.SRCQ.128.ST.INCP q%[qh], q7, %[dst]" "\n"

                    : 
                        [dst] "+r" (dst),
                        "=m" (*(uint8_t(*)[32])dst),
                        "+m" (dep)
                    :   
                        [ql] "n" (QL::N),
                        [qh] "n" (QH::N),
                        [off] "r" (off)
                );

                incp(dst,-16);
            }
            return dst;
        }

    }


}

