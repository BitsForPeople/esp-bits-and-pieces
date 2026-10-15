#pragma once
/*
 * Copyright 2026, <https://github.com/BitsForPeople>
 */
#include <cstdint>
#include <array>


namespace simd {
    struct QRs;

    struct QR_t {
        const unsigned n;

        constexpr auto operator <=>(const QR_t& other) const {
            return this->n <=> other.n;
        }

        constexpr auto operator <=>(const int i) const {
            return this->n <=> (unsigned)i;
        }

        constexpr operator unsigned(void) const {
            return n;
        }

        private:
        friend struct QRs;
        constexpr QR_t(const unsigned n) : n {n} {

        }
    };

    struct QRs {
        static constexpr std::array<QR_t,8> q = {
            0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u
        };
    };

    static constexpr const QR_t& Q0 = QRs::q[0];
    static constexpr const QR_t& Q1 = QRs::q[1];
    static constexpr const QR_t& Q2 = QRs::q[2];
    static constexpr const QR_t& Q3 = QRs::q[3];
    static constexpr const QR_t& Q4 = QRs::q[4];
    static constexpr const QR_t& Q5 = QRs::q[5];
    static constexpr const QR_t& Q6 = QRs::q[6];
    static constexpr const QR_t& Q7 = QRs::q[7];


    template<QR_t... Qs>
    inline constexpr bool noneOf(const QR_t& q) {
        return ((q != Qs) && ...);
    }

    template<QR_t... Qs>
    inline constexpr bool anyOf(const QR_t& q) {
        return ((q == Qs) || ...);
    }


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
            if constexpr (D != S) {
                asm (
                    "MV.QR q%[d], q%[s]"
                    :
                        "+m" (dep)
                    :
                        [d] "n" (D.n),
                        [s] "n" (S.n)
                );
            }
        }

        template<simd::QR_t Q, unsigned W>
        requires (W < 4)
        inline uint32_t getw(Dep& dep) {
            uint32_t data;
            asm (
                "EE.MOVI.32.A q%[qr], %[data], %[w]" "\n"
                : [data] "=r" (data)
                : [qr] "n" (Q.n),
                "m" (dep),
                [w] "n" (W)
            );
            return data;
        }

        template<simd::QR_t Q>
        inline void rotrq(const uint32_t bytes, Dep& dep) {
            asm (
                "WUR.SAR_BYTE %[s]" "\n"
                "EE.SRC.Q q%[qr], q%[qr], q%[qr]"
                : "+m" (dep)
                : [s] "r" (bytes),
                [qr] "n" (Q.n)
            );
        }


        template<simd::QR_t Q>
        inline void* storeQ(void* dst, Dep& dep) {
            if(known(((uintptr_t)dst % 16) == 0)) { // This check isn't worth doing at runtime.
                asm (
                    "EE.VST.128.IP q%[qr], %[dst], 16" "\n"
                    : [dst] "+r" (dst),
                    "=m" (*(uint8_t(*)[16])dst)
                    : [qr] "n" (Q.n),
                    "m" (dep)
                );
            } else
            if((uintptr_t)dst % 8 == 0) {
                asm (
                    "EE.VST.L.64.IP q%[qr], %[dst], 8" "\n"
                    "EE.VST.H.64.IP q%[qr], %[dst], 8" "\n"
                    : [dst] "+r" (dst),
                    "=m" (*(uint8_t(*)[16])dst)
                    : [qr] "n" (Q.n),
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
        requires (QL.n <= 5 && QH.n <= 5) // q6+q7 used as temporaries, clobbered
        inline void* store2Q(void* dst, Dep& dep) {

            if(known(((uintptr_t)dst % 16) == 0)) { 
                asm (
                    "EE.VST.128.IP q%[ql], %[dst], 16" "\n"
                    "EE.VST.128.IP q%[qh], %[dst], 16" "\n"
                    : [dst] "+r" (dst),
                        "=m" (*(uint8_t(*)[32])dst)
                    : "m" (dep),
                        [ql] "n" (QL.n),
                        [qh] "n" (QH.n)
                );
            } else
            if(((uintptr_t)dst % 8) == 0) {
                asm (
                    "EE.VST.L.64.IP q%[ql], %[dst], 8" "\n"
                    "EE.VST.H.64.IP q%[ql], %[dst], 8" "\n"
                    "EE.VST.L.64.IP q%[qh], %[dst], 8" "\n"
                    "EE.VST.H.64.IP q%[qh], %[dst], 8" "\n"
                    : [dst] "+r" (dst),
                    "=m" (*(uint8_t(*)[32])dst)
                    : [ql] "n" (QL.n),
                    [qh] "n" (QH.n),
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
                    :   [ql] "n" (QL.n),
                        [qh] "n" (QH.n),
                        [off] "r" (off)
                );

                incp(dst,-16);
            }
            return dst;
        }

    }


}

