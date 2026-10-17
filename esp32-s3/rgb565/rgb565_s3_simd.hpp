#pragma once
/*
 * Copyright 2026, <https://github.com/BitsForPeople>
 */
#include <cstdint>
#include "s3_simd_qtil.hpp"

namespace rgb565 {

    /**
     * @brief A single color/pixel in RGB565 format (B in lowest 5 bits, R in highest 5 bits)
     * 
     */
    struct Color565 {
        uint16_t val {};
        constexpr Color565(void) = default;
        constexpr Color565(const Color565& ) = default;
        constexpr Color565(const uint16_t val) : val {val} {

        }
        constexpr Color565(const uint8_t r, const uint8_t g, const uint8_t b) {
            this->set(r,g,b);
        }

        constexpr Color565& operator =(const uint16_t val) {
            this->val = val;
            return *this;
        }

        constexpr bool operator ==(const Color565& other) const {
            return this->val == other.val;
        }

        constexpr Color565& set(const uint8_t r, const uint8_t g, const uint8_t b) {
            this->val = ((uint16_t)(r & 0x1f) << 11) | ((uint16_t)(g & 0x3f) << 5) | (b & 0x1f);
            return *this;
        }

        constexpr uint8_t r(void) const {
            return this->val >> 11;
        }

        constexpr uint8_t g(void) const {
            return (this->val >> 5) & 0x3f;
        }
        constexpr uint8_t b(void) const {
            return (this->val) & 0x1f;
        }

    };

    /**
     * @brief A single color/pixel with separately stored R, G, and B values.
     * 
     */
    struct ColorRGB {
        union {
            std::array<uint8_t,3> rgb;
            struct {
                uint8_t r;
                uint8_t g;
                uint8_t b;
            };
        };
        constexpr ColorRGB(const ColorRGB&) = default;
        constexpr ColorRGB(void) {
            if(std::is_constant_evaluated()) {
                rgb = decltype(rgb) {};
            }
        }

        constexpr ColorRGB(const uint8_t r, const uint8_t g, const uint8_t b) :
            rgb {r,g,b} {

        }

        constexpr ColorRGB(const Color565& other) {
            this->from(other);
        }

        constexpr ColorRGB& from(const Color565& c) {
            this->rgb[0] = c.r();
            this->rgb[1] = c.g();
            this->rgb[2] = c.b();
            return *this;
        }

        constexpr Color565 as565(void) const {
            return Color565 (this->rgb[0], this->rgb[1], this->rgb[2]);
        }
    };


    template<unsigned FB = 8>
    requires (FB <= 8)
    struct FixedP {
        static constexpr uint8_t FRAC_BITS = FB;
        static constexpr uint8_t HALF = (FB != 0) ? ((uint8_t)(1u << (FB-1))) : 0;

        const uint8_t half {HALF}; // Keep this in RAM, close to val
        uint8_t val {};

        constexpr FixedP(void) = default;
        constexpr FixedP(const uint8_t val) : val {val} {

        }

        constexpr FixedP(const FixedP&) = default;
        constexpr FixedP& operator =(const FixedP&) = default;

        template<simd::QR_t Qhalf, simd::QR_t Qval>
        requires (Qhalf::N != Qval::N)
        void ld(simd::qtil::Dep& dep) const {
            simd::qtil::ldbc<Qhalf,uint8_t>(half,dep);
            simd::qtil::ldbc<Qval,uint8_t>(val,dep);
        }
    };


    /**
     * @brief A "vector" of 16 pixels, split into R, G, and B channels.
     * 
     * "Read" 16 RGB565 pixels via RGBVec::from565(const uint16_t*),
     * do vector-based (SIMD) operations on this "vector", then store the
     * result back as RGB565 via RGBVec::to565(void*).
     * 
     * Color components are stored as 5/6 bit values per channel, not extended
     * to 8 bits.
     * 
     * Because the PIE's EE.VMAX... and EE.VCMP... instructions work only on \e signed data,
     * RGBVec::clamp() uses \e signed 8-bit (S8), so the effective usable range for each
     * channel's intermediate values is only 7 bits (0...127). (Could be worked around at the
     * cost of some performance.)
     */
    class RGBVec {
        public:
        union {
            std::array<simd::Vec<int8_t>,3> rgb;
            struct {
                std::array<int8_t,16> r;
                std::array<int8_t,16> g;
                std::array<int8_t,16> b;
            };
        };

        constexpr RGBVec(void) {
            if(std::is_constant_evaluated()) {
                rgb = decltype(rgb) {};
            }
        };

        constexpr RGBVec(const RGBVec& other) {
            this->operator =(other);
        }

        constexpr RGBVec& operator =(const RGBVec& other) {
            if (std::is_constant_evaluated()) {
                this->rgb = other.rgb;
            } else {
                asm (
                    "LD.QR q0, %[src], 0" "\n"
                    "LD.QR q1, %[src], 16" "\n"
                    "LD.QR q2, %[src], 32" "\n"

                    "ST.QR q0, %[dst], 0" "\n"
                    "ST.QR q1, %[dst], 16" "\n"
                    "ST.QR q2, %[dst], 32" "\n"
                    :
                        "=m" (this->rgb)
                    :
                        [dst] "r" (&this->rgb),
                        [src] "r" (&other.rgb),
                        "m" (other.rgb)                  
                );
            }
            return *this;
        }

        constexpr ColorRGB operator[](const unsigned i) const {
            return ColorRGB (this->rgb[0][i], this->rgb[1][i], this->rgb[2][i]);
        }


        static void blend(RGBVec& dst,
                const RGBVec& src1,
                const uint8_t alpha1Q8,
                const RGBVec& src2,
                const uint8_t alpha2Q8) {

            constexpr uint32_t SHIFT = 8;

            simd::qtil::Dep dep;
            {
                uint8_t alphas[2];
                uint8_t* pa = &alphas[0];
                asm (
                    "S8I %[a1], %[pa], 0" "\n"
                    "S8I %[a2], %[pa], 1" "\n"
                    "EE.VLDBC.8.IP q6, %[pa], 1" "\n"
                    "EE.VLDBC.8 q7, %[pa]" "\n"
                    :
                        [pa] "+r" (pa),
                        "=m" (alphas),
                        "=m" (dep)
                    :
                        [a1] "r" (alpha1Q8),
                        [a2] "r" (alpha2Q8)
                );
            }

            const int8_t* s1 = src1.data();
            const int8_t* s2 = src2.data();

            uint32_t shift;
            asm (
                "EE.VLD.128.IP q0, %[src1], 16" "\n"
                "EE.VLD.128.IP q1, %[src2], 16" "\n"

                "EE.ZERO.QACC" "\n"
                "EE.VMULAS.U8.QACC.LD.IP q0, %[src1], 16, q0, q6" "\n"
                "EE.VMULAS.U8.QACC.LD.IP q1, %[src2], 16, q1, q7" "\n"

                "movi %[shift], %[SHIFT]" "\n" // free instruction (pipeline stall) 

                "EE.SRCMB.S8.QACC q2, %[shift], 0" "\n"

                "EE.ZERO.QACC" "\n"
                "EE.VMULAS.U8.QACC.LD.IP q0, %[src1], 16, q0, q6" "\n"
                "EE.VMULAS.U8.QACC.LD.IP q1, %[src2], 16, q1, q7" "\n"

                "ST.QR q2, %[dst], 0" "\n"

                "EE.SRCMB.S8.QACC q2, %[shift], 0" "\n"

                "EE.ZERO.QACC" "\n"
                "EE.VMULAS.U8.QACC q0, q6" "\n"
                "EE.VMULAS.U8.QACC q1, q7" "\n"

                "ST.QR q2, %[dst], 16" "\n"

                "EE.SRCMB.S8.QACC q2, %[shift], 0" "\n"

                "ST.QR q2, %[dst], 32" "\n"

                :
                    "=m" (dst.rgb),
                    "+m" (dep),
                    [src1] "+&r" (s1),
                    [src2] "+&r" (s2),
                    [shift] "=&r" (shift)
                : 
                    "m" (src1.rgb),
                    "m" (src2.rgb),
                    [dst] "r" (&dst.rgb),
                    [SHIFT] "n" (SHIFT)
            );

            [[assume(shift == SHIFT)]];
        }

        RGBVec& blend(const RGBVec& other, const uint8_t alphaQ8) {
            blend(*this,*this,256-alphaQ8,other,alphaQ8);
            return *this;
        }

        /**
         * @brief Scale color values from \p src by a factor of \p scaleFP / (1 << \p fracBits )
         * 
         * @param dst 
         * @param src 
         * @param scaleFP fixed-point scaling factor
         * @param fracBits number of fractional bits of \p scaleFP, 0...8
         */
        static inline void scale(RGBVec& dst, const RGBVec& src, const uint8_t scaleFP, const unsigned fracBits = 8) {
            [[assume(fracBits <= 8)]];
            simd::qtil::Dep dep;

            asm (
                "LD.QR q0, %[src], 0" "\n"
                "LD.QR q1, %[src], 16" "\n"
                "LD.QR q2, %[src], 32" "\n"
                :
                    "+m" (dep)
                : 
                    [src] "r" (&src.rgb),
                    "m" (src.rgb)
            );

            simd::qtil::ldbc<simd::Q7,uint8_t>(scaleFP, dep);

            asm (
                "WSR.SAR %[shift]" "\n"

                "EE.VMUL.U8 q0, q0, q7" "\n"
                "EE.VMUL.U8 q1, q1, q7" "\n"
                "EE.VMUL.U8 q2, q2, q7" "\n"

                "ST.QR q0, %[dst], 0" "\n"
                "ST.QR q1, %[dst], 16" "\n"
                "ST.QR q2, %[dst], 32" "\n"

                :
                    "=m" (dst.rgb),
                    "+m" (dep)
                : 
                    [src] "r" (&src.rgb),
                    "m" (src.rgb),
                    [dst] "r" (&dst.rgb),
                    [shift] "r" (fracBits)
            );

        }

        template<unsigned FB>
        static inline void scale_rnd(RGBVec& dst, const RGBVec& src, const FixedP<FB>& scale) {
            simd::qtil::Dep dep;

            using Qscale = simd::Q7;
            using Qhalf = simd::Q6;

            scale.template ld<Qhalf,Qscale>(dep);

            auto* s = &src.rgb;
            asm (
                "EE.VLD.128.IP q0, %[s], 16" "\n" // q0 <- r

                // QACC[n] <- 0.5 + (r[n]*scale)
                "EE.MOV.U8.QACC q%[half]" "\n" // QACC[n] <- 0.5
                "EE.VMULAS.U8.QACC.LD.IP q0, %[s], 16, q0, q%[scale]" "\n" // q0 <- g

                "EE.VLD.128.IP q1, %[s], -32" "\n" // q1 <- b

                "EE.SRCMB.S8.QACC q2, %[fracBits], 0" "\n" // q2 <- r'

                // QACC[n] <- 0.5 + (g[n]*scale)
                "EE.MOV.U8.QACC q%[half]" "\n" // QACC[n] <- 0.5
                "EE.VMULAS.U8.QACC q0, q%[scale]" "\n"

                "ST.QR q2, %[dst], 0" "\n" // r <- q2

                "EE.SRCMB.S8.QACC q2, %[fracBits], 0" "\n" // q2 <- g'

                // QACC[n] <- 0.5 + (b[n]*scale)
                "EE.MOV.U8.QACC q%[half]" "\n" // QACC[n] <- 0.5
                "EE.VMULAS.U8.QACC q1, q%[scale]" "\n"

                "ST.QR q2, %[dst], 16" "\n" // g <- q2

                "EE.SRCMB.S8.QACC q2, %[fracBits], 0" "\n" // q2 <- b'

                "ST.QR q2, %[dst], 32" "\n" // b <- q2
                :
                    "=m" (dst.rgb),
                    "+m" (dep),
                    [s] "+&r" (s)
                : 
                    "m" (src.rgb),
                    [dst] "r" (&dst.rgb),
                    [fracBits] "r" (scale.FRAC_BITS),
                    [scale] "n" (Qscale::N),
                    [half] "n" (Qhalf::N)
            );

            [[assume(s == &src.rgb)]];

        }


        inline RGBVec& scale(const uint8_t scaleFP, const unsigned fracBits = 8) {
            scale(*this,*this,scaleFP,fracBits);
            return *this;
        }

        inline RGBVec& scale_rnd(const uint8_t scaleQ8) {
            return scale_rnd(FixedP<8> {scaleQ8});
        }

        template<unsigned FB>
        inline RGBVec& scale_rnd(const FixedP<FB>& scale) {
            scale_rnd(*this,*this,scale);
            return *this;
        }

        static inline void add(RGBVec& dst, const RGBVec& src1, const RGBVec& src2) {
            simd::qtil::Dep dep {};

            simd::qtil::ldqr<simd::Q1>(&src1.rgb,dep); // q1 <- r
            simd::qtil::ldqr<simd::Q2>(&src2.rgb,dep); // q2 <- r

            asm (
                "EE.VADDS.S8 q0, q1, q2" "\n"
                
                "LD.QR q1, %[src1], 16" "\n" // q1 <- g
                "LD.QR q2, %[src2], 16" "\n" // q2 <- g

                "ST.QR q0, %[dst], 0" "\n" // r <- q0

                "EE.VADDS.S8 q0, q1, q2" "\n"

                "LD.QR q1, %[src1], 32" "\n" // q1 <- b
                "LD.QR q2, %[src2], 32" "\n" // q2 <- b

                "ST.QR q0, %[dst], 16" "\n" // g <- q0

                "EE.VADDS.S8 q0, q1, q2" "\n"

                "ST.QR q0, %[dst], 32" "\n" // b <- q0
                :
                    "=m" (dst.rgb),
                    "+m" (dep)
                :
                    "m" (src1.rgb),
                    "m" (src2.rgb),
                    [src1] "r" (&src1.rgb),
                    [src2] "r" (&src2.rgb),
                    [dst] "r" (&dst.rgb)
            );
        }

        static inline void add_clamp(RGBVec& dst, const RGBVec& src1, const RGBVec& src2) {
            simd::qtil::Dep dep {};

            simd::qtil::ldqr<simd::Q1>(&src1.rgb,dep); // q1 <- r
            simd::qtil::ldqr<simd::Q2>(&src2.rgb,dep); // q2 <- r

            loadMasks<simd::Q6,simd::Q7>(dep);

            asm (
                "EE.VADDS.S8 q0, q1, q2" "\n"
                
                "LD.QR q1, %[src1], 16" "\n" // q1 <- g
                "LD.QR q2, %[src2], 16" "\n" // q2 <- g

                "EE.VMIN.S8 q0, q0, q6" "\n" // q0 <- r'
                "ST.QR q0, %[dst], 0" "\n" // r <- q0

                "EE.VADDS.S8 q0, q1, q2" "\n"

                "LD.QR q1, %[src1], 32" "\n" // q1 <- b
                "LD.QR q2, %[src2], 32" "\n" // q2 <- b

                "EE.VMIN.S8 q0, q0, q7" "\n" // q0 <- g'
                "ST.QR q0, %[dst], 16" "\n" // g <- q0

                "EE.VADDS.S8 q0, q1, q2" "\n"
                "EE.VMIN.S8 q0, q0, q6" "\n" // q0 <- b'

                "ST.QR q0, %[dst], 32" "\n" // b <- q0
                :
                    "=m" (dst.rgb),
                    "+m" (dep)
                :
                    "m" (src1.rgb),
                    "m" (src2.rgb),
                    [src1] "r" (&src1.rgb),
                    [src2] "r" (&src2.rgb),
                    [dst] "r" (&dst.rgb)
            );
        }

        inline RGBVec& add(const RGBVec& src) {
            add(*this,*this,src);
            return *this;
        }

        inline RGBVec& operator +=(const RGBVec& other) {
            return add(other);
        }

        inline RGBVec& add_clamp(const RGBVec& other) {
            add_clamp(*this,*this,other);
            return *this;
        }


        /**
         * @brief Rotates "right"/down all 16 pixels from \p src by \p cnt (% 16) pixels
         * and stores the result in \p dst, basically:
         * <tt>dst[n] := src[(n+cnt) % 16]</tt>
         * 
         * @param dst 
         * @param src 
         * @param cnt 
         */
        static inline void rotr(RGBVec& dst, const RGBVec& src, const uint32_t cnt) {
            auto d = &dst.rgb;
            asm (
                "WUR.SAR_BYTE %[cnt]" "\n"

                "LD.QR q0, %[s], 0" "\n"
                "LD.QR q1, %[s], 16" "\n"
                "LD.QR q2, %[s], 32" "\n"

                "EE.SRCQ.128.ST.INCP q0, q0, %[d]" "\n"
                "EE.SRCQ.128.ST.INCP q1, q1, %[d]" "\n"
                "EE.SRCQ.128.ST.INCP q2, q2, %[d]" "\n"

                :
                    [d] "+r" (d),
                    "=m" (dst.rgb)
                :
                    [s] "r" (&src.rgb),
                    "m" (src.rgb),
                    [cnt] "r" (cnt)
            );

        }

        inline RGBVec& rotr(const uint32_t bytecnt) {
            rotr(*this,*this,bytecnt);
            return *this;
        }

        inline RGBVec& rotl(const uint32_t bytecnt) {
            return rotr(-bytecnt);
        }

        constexpr RGBVec& clear(void) {
            return this->setBlack();
        }

        constexpr RGBVec& setBlack(void) {
            return this->fill(0);
        }

        constexpr RGBVec& setWhite(void) {
            return this->fill(0xffffu);
        }

        constexpr RGBVec& fill(const ColorRGB& c) {
            if(std::is_constant_evaluated()) {
                std::fill(this->rgb[0].data.begin(), this->rgb[0].data.end(), c.rgb[0] );
                std::fill(this->rgb[1].data.begin(), this->rgb[1].data.end(), c.rgb[1] );
                std::fill(this->rgb[2].data.begin(), this->rgb[2].data.end(), c.rgb[2] );
            } else {
                const uint8_t* v = &c.rgb[0];
                asm (
                    "EE.VLDBC.8.IP q0, %[v], 1" "\n"
                    "EE.VLDBC.8.IP q1, %[v], 1" "\n"
                    "EE.VLDBC.8.IP q2, %[v], 1" "\n"

                    "ST.QR q0, %[rgb], 0" "\n"
                    "ST.QR q1, %[rgb], 16" "\n"
                    "ST.QR q2, %[rgb], 32" "\n"
                    :
                        "=m" (this->rgb),
                        [v] "+&r" (v)
                    :
                        [rgb] "r" (&this->rgb),
                        "m" (c.rgb)
                );
            }
            return *this;
        }

        constexpr RGBVec& fill(const Color565& c) {
            if(std::is_constant_evaluated()) {
                // const Color565 c {color};
                std::fill(this->rgb[0].data.begin(), this->rgb[0].data.end(), c.r() );
                std::fill(this->rgb[1].data.begin(), this->rgb[1].data.end(), c.g() );
                std::fill(this->rgb[2].data.begin(), this->rgb[2].data.end(), c.b() );
            } else {
                if(c.val == 0) {
                    asm (
                        "EE.ZERO.Q q0" "\n"
                    
                        "ST.QR q0, %[rgb], 0" "\n"
                        "ST.QR q0, %[rgb], 16" "\n"
                        "ST.QR q0, %[rgb], 32" "\n"
                        :
                            "=m" (this->rgb)
                        :
                            [rgb] "r" (&this->rgb)
                    );
                } else
                if(c.val == 0xffffu) {
                    asm (
                        "EE.ZERO.Q q0" "\n"
                        "EE.NOTQ q0, q0" "\n"
                    
                        "ST.QR q0, %[rgb], 0" "\n"
                        "ST.QR q0, %[rgb], 16" "\n"
                        "ST.QR q0, %[rgb], 32" "\n"
                        :
                            "=m" (this->rgb)
                        :
                            [rgb] "r" (&this->rgb)
                    );
                } else {
                    fill(ColorRGB {c});
                }
            }
            return *this;
        }

        constexpr RGBVec& from565(const uint16_t* const src) {
            if(std::is_constant_evaluated()) {
                for(unsigned i = 0; i < 16; ++i) {
                    const Color565 c {src[i]};
                    this->rgb[0][i] = c.r();
                    this->rgb[1][i] = c.g();
                    this->rgb[2][i] = c.b();
                }
            } else {
                simd::qtil::Dep dep {};

                using QL = simd::Q0;
                using QH = simd::Q1;
                using Qtmp = simd::Q2;

                if(known(((uintptr_t)src % 16) == 0)) {
                    simd::qtil::ldqr<QL,QH>(src,dep);
                } else {
                    const uint16_t* s = src;
                    simd::qtil::vldip_usar<QL,QH>(s,dep);
                    simd::qtil::vldip<Qtmp,-32>(s,dep);
                    [[assume(s == src)]];
                    asm (
                        "EE.SRC.Q q%[ql], q%[ql], q%[qh]" "\n"
                        "EE.SRC.Q q%[qh], q%[qh], q%[tmp]" "\n"
                        :
                            "+m" (dep)
                        :
                            [ql] "n" (QL::N),
                            [qh] "n" (QH::N),
                            [tmp] "n" (Qtmp::N)
                    );

                }

                from565<QL,QH>(dep);

            }
            return *this;
        }


        constexpr RGBVec& from565Masked(const uint16_t* src, const uint16_t keyColor) {
            if(std::is_constant_evaluated()) {
                for(unsigned i = 0; i < 16; ++i) {
                    const Color565 c {src[i]};
                    if(c.val != keyColor) {
                        this->rgb[0][i] = c.r();
                        this->rgb[1][i] = c.g();
                        this->rgb[2][i] = c.b();
                    } else {
                        this->rgb[0][i] = 0;
                        this->rgb[1][i] = 0;
                        this->rgb[2][i] = 0;
                    }
                }
            } else {
                simd::qtil::Dep dep {};

                using QL = simd::Q0;
                using QH = simd::Q1;
                using Qtmp = simd::Q2;
                using Qkey = simd::Q7;

                simd::qtil::ldbc<Qkey,uint16_t>(keyColor, dep);

                if(known(((uintptr_t)src % 16) == 0)) {
                    simd::qtil::ldqr<QL,QH>(src,dep);
                } else {
                    const uint16_t* s = src;
                    simd::qtil::vldip_usar<QL,QH>(s,dep);
                    simd::qtil::vldip<Qtmp,-32>(s,dep);
                    [[assume(s == src)]];

                    asm (
                        "EE.SRC.Q q%[ql], q%[ql], q%[qh]" "\n"
                        "EE.SRC.Q q%[qh], q%[qh], q%[tmp]" "\n"
                        :
                            "+m" (dep)
                        :
                            [ql] "n" (QL::N),
                            [qh] "n" (QH::N),
                            [tmp] "n" (Qtmp::N)
                    );

                }

                asm (
                    "EE.VCMP.EQ.S16 q%[tmp], q%[ql], q%[key]" "\n"
                    "EE.NOTQ q%[tmp], q%[tmp]" "\n"
                    "EE.ANDQ q%[ql], q%[ql], q%[tmp]" "\n"

                    "EE.VCMP.EQ.S16 q%[tmp], q%[qh], q%[key]" "\n"
                    "EE.NOTQ q%[tmp], q%[tmp]" "\n"
                    "EE.ANDQ q%[qh], q%[qh], q%[tmp]" "\n"
                    :
                        "+m" (dep)
                    :
                        [ql] "n" (QL::N),
                        [qh] "n" (QH::N),
                        [tmp] "n" (Qtmp::N),
                        [key] "n" (Qkey::N)
                );

                from565<QL,QH>(dep);

            }
            return *this;
        }


        /**
         * @brief Convert to and store the 16 RGB565 pixels (32 bytes) to \p dst.
         *
         * @note This uses 16-byte aligned reads/writes even on unaligned \p dst.
         * I.e., data up to 15 bytes below *dst or up to 15 bytes above *(dst+32)
         * may be read and re-written. That is not a problem \e unless data in that memory
         * may be modified concurrently from another task or ISR. Make sure this cannot happen,
         * e.g. by aligning or padding the dst buffer to 16 bytes.
         * 
         * @tparam BYTESWAP \c true to swap low/high bytes; default: \c false
         * @param dst 
         */
        template<bool BYTESWAP = false>
        constexpr void to565(void* dst) const {
            if(std::is_constant_evaluated()) {
                uint16_t* const d = (uint16_t*)dst;
                for(unsigned i = 0; i < 16; ++i) {
                    const Color565 c (this->rgb[0][i], this->rgb[1][i], this->rgb[2][i]);

                    uint32_t v = c.val;
                    if constexpr (BYTESWAP) {
                        v = (v >> 8) | (v<<8);
                    }

                    d[i] = v;
                }
            } else {
                using QL = simd::Q0;
                using QH = simd::Q1;

                simd::qtil::Dep dep {};

                to565<QL,QH,BYTESWAP>(dep);
                simd::qtil::store2Q<QL,QH>(dst, dep);
            }
        }

        /**
         * @brief Same as RGBVec::to565(void*), but only stores the first \p cnt (<=16)
         * pixels to \p dst . 
         * 
         * @tparam BYTESWAP \c true to swap low/high bytes; default: \c false
         * @param dst 
         * @param cnt 
         * @see RGBVec::to565(void*)
         */
        template<bool BYTESWAP = false>
        constexpr void to565(void* dst, uint32_t cnt) const {
            if(std::is_constant_evaluated()) {
                uint16_t* const d = (uint16_t*)dst;
                cnt = cnt < 16 ? cnt : 16;
                for(unsigned i = 0; i < cnt; ++i) {
                    const Color565 c (this->rgb[0][i], this->rgb[1][i], this->rgb[2][i]);
                    uint32_t v = c.val;
                    if constexpr (BYTESWAP) {
                        v = (v >> 8) | (v<<8);
                    }
                    d[i] = v;
                }
            } else {

                using QL = simd::Q0;
                using QH = simd::Q1;

                simd::qtil::Dep dep {};

                to565<QL,QH,BYTESWAP>(dep);

                if(cnt < 16) [[likely]] {
                    if(cnt & 8) {
                        dst = simd::qtil::storeQ<QL>(dst,dep);
                        cnt -= 8;
                        simd::qtil::mv<QL,QH>(dep);
                    }
                    if(cnt) {
                        simd::qtil::storeQ<QL>(dst,cnt*sizeof(uint16_t),dep);
                    }
                } else {
                    simd::qtil::store2Q<QL,QH>(dst, dep);
                }
            }
        }

        /**
         * @brief Clamp the current R, G, and B values to their respective RGB565 max values (0x1f/0x3f).
         * 
         * @return RGBVec& 
         */
        RGBVec& clamp(void) {
            simd::qtil::Dep dep {};

            using M5 = simd::Q6;
            using M6 = simd::Q7;

            using R = simd::Q0;
            using G = simd::Q1;
            using B = simd::Q2;

            loadMasks<M5,M6>(dep);

            simd::qtil::ldqr<R,G,B>(&this->rgb,dep);

            asm (
                "EE.VMIN.S8 q%[r], q%[r], q%[m5]" "\n" // r
                "ST.QR q%[r], %[rgb], 0" "\n"

                "EE.VMIN.S8 q%[g], q%[g], q%[m6]" "\n" // g
                "ST.QR q%[g], %[rgb], 16" "\n"

                "EE.VMIN.S8 q%[b], q%[b], q%[m5]" "\n" // b
                "ST.QR q%[b], %[rgb], 32" "\n"
                :
                    "=m" (this->rgb),
                    "+m" (dep)
                :
                    [rgb] "r" (&this->rgb),
                    [r] "n" (R::N),
                    [g] "n" (G::N),
                    [b] "n" (B::N),
                    [m5] "n" (M5::N),
                    [m6] "n" (M6::N)

            );

            return *this;
        }

        private:

        static inline constexpr uint16_t MASK_5 = (1u << 5) - 1; // 0x001f;
        static inline constexpr uint16_t MASK_6 = (1u << 6) - 1; // 0x003f;

        static inline constexpr uint16_t MASKS[2] = {MASK_5, MASK_6};



        constexpr int8_t* data(void) {
            return &this->rgb[0][0];
        }

        constexpr const int8_t* data(void) const {
            return &this->rgb[0][0];
        }

        template<simd::QR_t M5 = simd::Q6, simd::QR_t M6 = simd::Q7>
        requires (M5::N != M6::N)
        static inline void loadMasks(simd::qtil::Dep& dep) {
            const auto* masks = &MASKS[0];
            simd::qtil::ldbcip<M5,uint8_t,sizeof(MASKS[0])>(masks,dep);
            simd::qtil::ldbc<M6,uint8_t>(*masks,dep);
        }


        /** Input pixels in (QH:QL)
         * 
         */
        template<
            simd::qr_noneOf<simd::Q2,simd::Q3,simd::Q6,simd::Q7> QL,
            simd::qr_noneOf<simd::Q2,simd::Q3,simd::Q6,simd::Q7> QH
            >
        requires (QL::N != QH::N)
        inline void from565(simd::qtil::Dep& dep) {
            loadMasks<simd::Q6,simd::Q7>(dep);
            asm (
                "SSAI %[shift]" "\n"

                "EE.VSL.32 q2, q%[ql]" "\n" // left-align B (even bytes), right-align G (odd bytes)
                "EE.VSL.32 q3, q%[qh]" "\n"

                "EE.VUNZIP.8 q2, q3" "\n" // (q2 <- left-aligned B), q3 <- right-aligned G
                "EE.VUNZIP.8 q%[ql], q%[qh]" "\n" //  qh <- left-aligned R , ql <- right-aligned B

                "EE.VSR.32 q%[qh], q%[qh]" "\n" // right-align R
                "EE.ANDQ q%[qh], q%[qh], q6" "\n" // VSR.32 sign-extends
                "ST.QR q%[qh], %[rgb], 0" "\n" // red done

                "EE.ANDQ q3, q3, q7" "\n" // mask off the lowest 2 bits of R from G
                "ST.QR q3, %[rgb], 16" "\n" // green done 

                "EE.ANDQ q%[ql], q%[ql], q6" "\n" // mask off the lowest 3 bits of G from B
                "ST.QR q%[ql], %[rgb], 32" "\n" // blue done

                :
                    "+m" (dep),
                    "=m" (this->rgb)
                :
                    [rgb] "r" (&this->rgb),
                    [shift] "n" (8-5),
                    [ql] "n" (QL::N),
                    [qh] "n" (QH::N)
            );
        }


        template<simd::qr_noneOf<simd::Q1,simd::Q3> QL, simd::QR_t QH, bool BYTESWAP = false>
        inline void to565(simd::qtil::Dep& dep) const {
            asm (
                "LD.QR q3, %[rgb], 0" "\n" // r
                "LD.QR q1, %[rgb], 16" "\n" // g
                "LD.QR q0, %[rgb], 32" "\n" // b

                "SSAI %[shift]" "\n"
                "EE.VSL.32 q0, q0" "\n" // left-align the 5 B bits
                "EE.VZIP.8 q0, q1" "\n" // merge B+G into 16 bits

                "EE.VSR.32 q0, q0" "\n" // shift B+G back down
                "EE.VSR.32 q1, q1" "\n"

                "EE.VSL.32 q3, q3" "\n" // left-align the 5 R bits
                "EE.ZERO.Q q2" "\n"
                "EE.VZIP.8 q2, q3" "\n" // implicit left-shift by 8

                "EE.ORQ q%[ql], q0, q2" "\n"
                "EE.ORQ q%[qh], q1, q3" "\n"
                
                :
                    "+m" (dep)
                :
                    [rgb] "r" (&this->rgb),
                    "m" (this->rgb),
                    [shift] "n" (8-5),
                    [ql] "n" (QL::N),
                    [qh] "n" (QH::N)
            );
            if constexpr (BYTESWAP) {
                simd::qtil::swap8<QL,QH>(dep);
            }
        }

        static constexpr bool known(const bool c) {
            return __builtin_constant_p(c) && c;
        }

        static constexpr bool maybe(const bool c) {
            return !known(!c);
        }
    };


}