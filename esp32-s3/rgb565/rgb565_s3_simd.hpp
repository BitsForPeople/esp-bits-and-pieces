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
            alignas(16) std::array<std::array<int8_t,16>,3> rgb;
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
                "EE.VLDBC.8 q7, %[scale]" 
                :
                    "+m" (dep)
                :
                    [scale] "r" (&scaleFP),
                    "m" (scaleFP)
            );

            asm (
                "LD.QR q0, %[src], 0" "\n"
                "LD.QR q1, %[src], 16" "\n"
                "LD.QR q2, %[src], 32" "\n"

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

        /**
         * @brief Scale color values from \p src by a factor of \p scaleFP / (1 << \p fracBits ) with rounding
         * (nearest-integer) of the results.
         * 
         * @param dst 
         * @param src 
         * @param scaleFP fixed-point scaling factor
         * @param fracBits number of fractional bits of \p scaleFP, 0...8
         */
        static inline void scale_rnd(RGBVec& dst, const RGBVec& src, const uint8_t scaleFP, const unsigned fracBits = 8) {
            [[assume(fracBits <= 8)]];
            simd::qtil::Dep dep;
            // q7[n] := scaleFP
            asm (
                "EE.ZERO.Q q6" "\n"
                "EE.VLDBC.8 q7, %[scale]" "\n"
                :
                    "+m" (dep)
                :
                    [scale] "r" (&scaleFP),
                    "m" (scaleFP)
            );

            if(fracBits != 0) {
                // set up q6 with 0.5 for the rounding
                // q6[n] := 0.5
                const uint8_t half = 1u << (fracBits-1);
                asm (
                    "EE.VLDBC.8 q6, %[half]" "\n"
                    : 
                        "+m" (dep)
                    : 
                        [half] "r" (&half),
                        "m" (half)
                );
            }



            auto s = &src.rgb;
            asm (
                "EE.VLD.128.IP q0, %[s], 16" "\n" // q0 <- r

                // QACC[n] <- 0.5 + ((r[n]*scaleFP) >> fracBits)
                "EE.MOV.U8.QACC q6" "\n" // QACC[n] <- 0.5
                "EE.VMULAS.U8.QACC.LD.IP q0, %[s], 16, q0, q7" "\n" // q0 <- g

                "EE.VLD.128.IP q1, %[s], -32" "\n" // q1 <- b

                "EE.SRCMB.S8.QACC q2, %[fracBits], 0" "\n" // q2[n] <- QACC[n] >> fracBits

                // QACC[n] <- 0.5 + ((g[n]*scaleFP) >> fracBits)
                "EE.MOV.U8.QACC q6" "\n" // QACC[n] <- 0.5
                "EE.VMULAS.U8.QACC q0, q7" "\n"

                "ST.QR q2, %[dst], 0" "\n" // r <- q2

                "EE.SRCMB.S8.QACC q2, %[fracBits], 0" "\n"

                // QACC[n] <- 0.5 + ((b[n]*scaleFP) >> fracBits)
                "EE.MOV.U8.QACC q6" "\n" // QACC[n] <- 0.5
                "EE.VMULAS.U8.QACC q1, q7" "\n"

                "ST.QR q2, %[dst], 16" "\n" // g <- q2

                "EE.SRCMB.S8.QACC q2, %[fracBits], 0" "\n"

                "ST.QR q2, %[dst], 32" "\n" // b <- q2
                :
                    "=m" (dst.rgb),
                    "+m" (dep),
                    [s] "+&r" (s)
                : 
                    "m" (src.rgb),
                    [dst] "r" (&dst.rgb),
                    [fracBits] "r" (fracBits)
            );

            [[assume(s == &src.rgb)]];

        }

        inline RGBVec& scale(const uint8_t scaleFP, const unsigned fracBits = 8) {
            scale(*this,*this,scaleFP,fracBits);
            return *this;
        }

        inline RGBVec& scale_rnd(const uint8_t scaleFP, const unsigned fracBits = 8) {
            scale_rnd(*this,*this,scaleFP,fracBits);
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
                std::fill(this->rgb[0].begin(), this->rgb[0].end(), c.rgb[0] );
                std::fill(this->rgb[1].begin(), this->rgb[1].end(), c.rgb[1] );
                std::fill(this->rgb[2].begin(), this->rgb[2].end(), c.rgb[2] );
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
                std::fill(this->rgb[0].begin(), this->rgb[0].end(), c.r() );
                std::fill(this->rgb[1].begin(), this->rgb[1].end(), c.g() );
                std::fill(this->rgb[2].begin(), this->rgb[2].end(), c.b() );
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

        constexpr RGBVec& from565(const uint16_t* src) {
            if(std::is_constant_evaluated()) {
                for(unsigned i = 0; i < 16; ++i) {
                    const Color565 c {src[i]};
                    this->rgb[0][i] = c.r();
                    this->rgb[1][i] = c.g();
                    this->rgb[2][i] = c.b();
                }
            } else {
                simd::qtil::Dep dep {};

                loadMasks<simd::Q6,simd::Q7>(dep);

                if(known(((uintptr_t)src % 16) == 0)) {
                    asm (
                        "LD.QR q0, %[src], 0" "\n"
                        "LD.QR q1, %[src], 16" "\n"
                        :
                            "+m" (dep)
                        :
                            [src] "r" (src),
                            "m" (*(const uint16_t(*)[16])src)
                    );
                } else {
                    asm (
                        "EE.LD.128.USAR.IP q0, %[src], 16" "\n"
                        "EE.VLD.128.IP q1, %[src], 16" "\n"
                        "EE.VLD.128.IP q2, %[src], 0" "\n"

                        "EE.SRC.Q q0, q0, q1" "\n"
                        "EE.SRC.Q q1, q1, q2" "\n"
                        :
                            [src] "+r" (src),
                            "+m" (dep)
                        :
                            "m" (*(const uint16_t(*)[16])src)
                    );
                }

                asm (
                    "SSAI %[shift]" "\n"

                    "EE.VSL.32 q2, q0" "\n" // left-align B (even bytes), right-align G (odd bytes)
                    "EE.VSL.32 q3, q1" "\n"

                    "EE.VUNZIP.8 q2, q3" "\n" // (q2 <- left-aligned B), q3 <- right-aligned G
                    "EE.VUNZIP.8 q0, q1" "\n" //  q1 <- left-aligned R , q0 <- right-aligned B

                    "EE.VSR.32 q1, q1" "\n" // right-align R
                    "EE.ANDQ q1, q1, q6" "\n" // VSR.32 sign-extends
                    "ST.QR q1, %[rgb], 0" "\n" // red done

                    "EE.ANDQ q3, q3, q7" "\n" // mask off the lowest 2 bits of R from G
                    "ST.QR q3, %[rgb], 16" "\n" // green done 

                    "EE.ANDQ q0, q0, q6" "\n" // mask off the lowest 3 bits of G from B
                    "ST.QR q0, %[rgb], 32" "\n" // blue done

                    :
                        "+m" (dep),
                        "=m" (this->rgb)
                    :
                        [rgb] "r" (&this->rgb),
                        [shift] "n" (8-5)              
                );
            }
            return *this;
        }

        /**
         * @brief Convert to and store the 16 RGB565 pixels to \p dst
         * 
         * @param dst 
         */
        constexpr void to565(void* dst) const {
            if(std::is_constant_evaluated()) {
                uint16_t* const d = (uint16_t*)dst;
                for(unsigned i = 0; i < 16; ++i) {
                    const Color565 c (this->rgb[0][i], this->rgb[1][i], this->rgb[2][i]);
                    d[i] = c.val;
                }
            } else {
                constexpr auto& QL = simd::Q0;
                constexpr auto& QH = simd::Q1;
                simd::qtil::Dep dep {};
                to565<QL,QH>(dep);
                simd::qtil::store2Q<QL,QH>(dst, dep);
            }
        }

        /**
         * @brief Same as RGBVec::to565(void*), but only stores the first \p cnt (<=16)
         * pixels to \p dst . 
         * 
         * @param dst 
         * @param cnt 
         * @see RGBVec::to565(void*)
         */
        constexpr void to565(void* dst, uint32_t cnt) const {
            simd::qtil::Dep dep {};
            constexpr auto& QL = simd::Q0;
            constexpr auto& QH = simd::Q1;
            to565<QL,QH>(dep);
            if(cnt >= 16) {
                simd::qtil::store2Q<QL,QH>(dst, dep);
            } else {
                if(cnt & 8) {
                    dst = simd::qtil::storeQ<QL>(dst,dep);
                    cnt -= 8;
                    simd::qtil::mv<QL,QH>(dep);
                }
                if(cnt) {
                    simd::qtil::storeQ<QL>(dst,cnt*sizeof(uint16_t),dep);
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

            loadMasks<simd::Q6,simd::Q7>(dep);

            asm (
                "LD.QR q0, %[rgb], 0" "\n" // r
                "LD.QR q1, %[rgb], 16" "\n" // g
                "LD.QR q2, %[rgb], 32" "\n" // b

                "EE.VMIN.S8 q0, q0, q6" "\n" // r
                "ST.QR q0, %[rgb], 0" "\n"

                "EE.VMIN.S8 q1, q1, q7" "\n" // g
                "ST.QR q1, %[rgb], 16" "\n"

                "EE.VMIN.S8 q2, q2, q6" "\n" // b
                "ST.QR q2, %[rgb], 32" "\n"
                :
                    "+m" (this->rgb),
                    "+m" (dep)
                :
                    [rgb] "r" (&this->rgb)
            );

            return *this;
        }

        private:

        static inline constexpr uint16_t MASK_5 = (1u << 5) - 1; // 0x1f;
        static inline constexpr uint16_t MASK_6 = (1u << 6) - 1; // 0x31;

        static inline constexpr uint16_t MASKS[2] = {MASK_5, MASK_6};



        constexpr int8_t* data(void) {
            return &this->rgb[0][0];
        }

        constexpr const int8_t* data(void) const {
            return &this->rgb[0][0];
        }

        template<simd::QR_t M5Q = simd::Q6, simd::QR_t M6Q = simd::Q7>
        requires (M5Q != M6Q)
        static inline void loadMasks(simd::qtil::Dep& dep) {
            const uint16_t* masks = &MASKS[0];
            asm (
                "EE.VLDBC.8.IP q%[m5], %[m], %[inc]" "\n" // q6[n] <- 0x1f (MASK_5)
                "EE.VLDBC.8    q%[m6], %[m]" "\n" // q7[n] <- 0x3f (MASK_6)
                :
                    [m] "+r" (masks),
                    "+m" (dep)
                :
                    "m" (MASKS),
                    [m5] "n" (M5Q.n),
                    [m6] "n" (M6Q.n),
                    [inc] "n" (sizeof(*masks))
            );
        }

        template<simd::QR_t QL, simd::QR_t QH>
        requires (simd::noneOf<simd::Q1,simd::Q3>(QL))
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
                    [ql] "n" (QL.n),
                    [qh] "n" (QH.n)
            );
        }

        static constexpr bool known(const bool c) {
            return __builtin_constant_p(c) && c;
        }

        static constexpr bool maybe(const bool c) {
            return !known(!c);
        }
    };


}