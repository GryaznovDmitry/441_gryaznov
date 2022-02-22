﻿// <auto-generated />
using System;
using Lab2;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Migrations;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;

#nullable disable

namespace Lab2.Migrations
{
    [DbContext(typeof(DetectedObjectContext))]
    [Migration("20220221023525_firstversion")]
    partial class firstversion
    {
        protected override void BuildTargetModel(ModelBuilder modelBuilder)
        {
#pragma warning disable 612, 618
            modelBuilder.HasAnnotation("ProductVersion", "6.0.2");

            modelBuilder.Entity("ModelLibrary.DetectedObject", b =>
                {
                    b.Property<int>("DetectedObjectId")
                        .ValueGeneratedOnAdd()
                        .HasColumnType("INTEGER");

                    b.Property<byte[]>("BitmapImageFull")
                        .IsRequired()
                        .HasColumnType("BLOB");

                    b.Property<byte[]>("BitmapImageObj")
                        .IsRequired()
                        .HasColumnType("BLOB");

                    b.Property<string>("Type")
                        .IsRequired()
                        .HasColumnType("TEXT");

                    b.Property<float>("x1")
                        .HasColumnType("REAL");

                    b.Property<float>("x2")
                        .HasColumnType("REAL");

                    b.Property<float>("y1")
                        .HasColumnType("REAL");

                    b.Property<float>("y2")
                        .HasColumnType("REAL");

                    b.HasKey("DetectedObjectId");

                    b.ToTable("DetectedObjects");
                });
#pragma warning restore 612, 618
        }
    }
}
